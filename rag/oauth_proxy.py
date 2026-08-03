"""
Google OAuth *proxy* authorization server for the MCP server (HTTP transport).

Google does not support OAuth 2.0 Dynamic Client Registration (DCR, RFC 7591),
so MCP clients (Claude Code, the claude.ai connector) cannot register with it
automatically — they dead-end on "does not support dynamic client registration"
and require a hand-configured client id + secret.

This module makes *this* server the authorization server the clients talk to,
and has it broker a second OAuth exchange with Google behind the scenes:

    +--------+     +------------+     +-------------------+
    | MCP    | --> | this MCP   | --> | Google (accounts. |
    | client |     | server     |     | google.com)       |
    +--------+     +------------+     +-------------------+

The MCP client speaks standard DCR + authorization-code + PKCE to us; we speak
a single pre-registered OAuth client to Google. The client never needs a Google
client id/secret — only our URL.

Token model: we mint our own opaque access/refresh tokens and keep the Google
tokens server-side, so Google credentials are never handed to clients. Access is
still gated to the configured Google Workspace domain / email allowlist, reusing
``rag.auth.GoogleTokenVerifier`` to validate the Google token at login time.

State is in-memory: a single server instance. A restart invalidates in-flight
logins and issued tokens, and clients re-authenticate — acceptable for a single
Railway instance. Move these stores to Postgres if the server is ever scaled out.
"""

import secrets
import time

import httpx
from mcp.server.auth.provider import (
    AccessToken,
    AuthorizationCode,
    AuthorizationParams,
    RefreshToken,
    TokenError,
    construct_redirect_uri,
)
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response

from rag.auth import GoogleTokenVerifier

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"

# Scopes we request from Google. "openid email" is the minimum the token
# verifier needs (verified email + domain gating).
GOOGLE_SCOPES = "openid email"

AUTH_CODE_TTL_SECONDS = 300  # our authorization codes are short-lived


class GoogleOAuthProxyProvider:
    """OAuthAuthorizationServerProvider that proxies to Google.

    Implements the MCP SDK's ``OAuthAuthorizationServerProvider`` protocol so it
    can be passed to ``FastMCP(auth_server_provider=...)``. The paired Google
    redirect handler is :meth:`handle_google_callback`, mounted as a custom route.
    """

    def __init__(
        self,
        *,
        public_url: str,
        google_client_id: str,
        google_client_secret: str,
        callback_path: str = "/auth/google/callback",
        allowed_domain: str = "",
        allowed_emails: list[str] | None = None,
    ) -> None:
        if not google_client_id or not google_client_secret:
            raise ValueError(
                "GoogleOAuthProxyProvider requires GOOGLE_OAUTH_CLIENT_ID and "
                "GOOGLE_OAUTH_CLIENT_SECRET (the server-side Google web client "
                "used to broker sign-in)."
            )
        self.public_url = public_url.rstrip("/")
        self.google_client_id = google_client_id
        self.google_client_secret = google_client_secret
        self.callback_path = callback_path
        self.redirect_uri = f"{self.public_url}{callback_path}"
        # Reuse the existing verifier for domain/email/audience gating. The
        # Google token's audience is our own client, so bind to it.
        self._verifier = GoogleTokenVerifier(
            allowed_domain=allowed_domain,
            allowed_emails=allowed_emails or None,
            allowed_client_ids=[google_client_id],
        )

        # In-memory state (single instance; see module docstring).
        self._clients: dict[str, OAuthClientInformationFull] = {}
        self._transactions: dict[str, dict] = {}  # google-state -> pending login
        self._auth_codes: dict[str, AuthorizationCode] = {}  # our code -> code
        self._code_google: dict[str, dict] = {}  # our code -> google token bundle
        self._access_tokens: dict[str, AccessToken] = {}  # our token -> AccessToken
        self._refresh_tokens: dict[str, RefreshToken] = {}  # our token -> RefreshToken
        self._refresh_google: dict[str, str] = {}  # our refresh -> google refresh

    # ── Dynamic client registration ─────────────────────────────────────────

    async def get_client(self, client_id: str) -> OAuthClientInformationFull | None:
        return self._clients.get(client_id)

    async def register_client(self, client_info: OAuthClientInformationFull) -> None:
        # The SDK's /register handler mints client_id/secret; we just persist.
        self._clients[client_info.client_id] = client_info

    # ── Authorization: redirect the user to Google ──────────────────────────

    async def authorize(
        self, client: OAuthClientInformationFull, params: AuthorizationParams
    ) -> str:
        # Open a login transaction keyed by the state we hand to Google, holding
        # everything needed to complete the client's leg after Google returns.
        google_state = secrets.token_urlsafe(32)
        self._transactions[google_state] = {
            "client_id": client.client_id,
            "redirect_uri": str(params.redirect_uri),
            "redirect_uri_provided_explicitly": params.redirect_uri_provided_explicitly,
            "client_state": params.state,
            "code_challenge": params.code_challenge,
            "scopes": params.scopes or ["openid", "email"],
            "resource": params.resource,
            "created_at": time.time(),
        }

        query = {
            "client_id": self.google_client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": GOOGLE_SCOPES,
            "state": google_state,
            "access_type": "offline",
            "prompt": "consent",
        }
        return construct_redirect_uri(GOOGLE_AUTH_URL, **query)

    async def handle_google_callback(self, request: Request) -> Response:
        """Google redirects here after the user signs in.

        Mounted as a custom GET route at ``self.callback_path``. Exchanges the
        Google code for tokens, gates access, mints our own authorization code,
        and redirects back to the MCP client's ``redirect_uri``.
        """
        params = request.query_params
        google_state = params.get("state", "")
        txn = self._transactions.pop(google_state, None)
        if txn is None:
            return PlainTextResponse("Invalid or expired login state", status_code=400)

        if params.get("error"):
            return self._fail_back(
                txn,
                params.get("error", "access_denied"),
                params.get("error_description"),
            )

        code = params.get("code")
        if not code:
            return self._fail_back(txn, "invalid_request", "missing authorization code")

        # Exchange the Google code for tokens (server-side, confidential client).
        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "code": code,
                        "client_id": self.google_client_id,
                        "client_secret": self.google_client_secret,
                        "redirect_uri": self.redirect_uri,
                        "grant_type": "authorization_code",
                    },
                )
        except Exception:
            return self._fail_back(txn, "server_error", "google token exchange failed")

        if resp.status_code != 200:
            return self._fail_back(
                txn, "access_denied", "google rejected the authorization"
            )

        google = resp.json()
        google_access = google.get("access_token", "")

        # Gate access: verify the Google token (verified email, domain/email
        # allowlist, audience bound to our client). Reuses rag.auth.
        access = await self._verifier.verify_token(google_access)
        if access is None:
            return self._fail_back(
                txn, "access_denied", "account not permitted for this server"
            )

        # Mint our own authorization code and stash the Google token bundle.
        our_code = secrets.token_urlsafe(32)
        self._auth_codes[our_code] = AuthorizationCode(
            code=our_code,
            scopes=txn["scopes"],
            expires_at=time.time() + AUTH_CODE_TTL_SECONDS,
            client_id=txn["client_id"],
            code_challenge=txn["code_challenge"],
            redirect_uri=AnyUrl(txn["redirect_uri"]),
            redirect_uri_provided_explicitly=txn["redirect_uri_provided_explicitly"],
            resource=txn["resource"],
        )
        self._code_google[our_code] = {
            "access_token": google_access,
            "refresh_token": google.get("refresh_token"),
            "expires_at": time.time() + int(google.get("expires_in", 3600)),
            "email": access.client_id,  # GoogleTokenVerifier sets client_id=email
        }

        return RedirectResponse(
            url=construct_redirect_uri(
                txn["redirect_uri"], code=our_code, state=txn["client_state"]
            ),
            status_code=302,
        )

    def _fail_back(self, txn: dict, error: str, description: str | None) -> Response:
        """Redirect the OAuth error back to the MCP client's redirect_uri."""
        return RedirectResponse(
            url=construct_redirect_uri(
                txn["redirect_uri"],
                error=error,
                error_description=description,
                state=txn["client_state"],
            ),
            status_code=302,
        )

    # ── Token endpoint ──────────────────────────────────────────────────────

    async def load_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: str
    ) -> AuthorizationCode | None:
        code = self._auth_codes.get(authorization_code)
        if code is None or code.client_id != client.client_id:
            return None
        if code.expires_at < time.time():
            self._auth_codes.pop(authorization_code, None)
            self._code_google.pop(authorization_code, None)
            return None
        return code

    async def exchange_authorization_code(
        self, client: OAuthClientInformationFull, authorization_code: AuthorizationCode
    ) -> OAuthToken:
        # Single-use: consume the code and its Google bundle.
        self._auth_codes.pop(authorization_code.code, None)
        bundle = self._code_google.pop(authorization_code.code, None)
        if bundle is None:
            raise TokenError(
                "invalid_grant", "authorization code already used or expired"
            )

        return self._issue_tokens(
            client_id=client.client_id,
            scopes=authorization_code.scopes,
            resource=authorization_code.resource,
            google_access=bundle["access_token"],
            google_refresh=bundle.get("refresh_token"),
            expires_at=bundle["expires_at"],
        )

    def _issue_tokens(
        self,
        *,
        client_id: str,
        scopes: list[str],
        resource: str | None,
        google_access: str,
        google_refresh: str | None,
        expires_at: float,
    ) -> OAuthToken:
        our_access = secrets.token_urlsafe(32)
        self._access_tokens[our_access] = AccessToken(
            token=our_access,
            client_id=client_id,
            scopes=scopes,
            expires_at=int(expires_at),
            resource=resource,
        )

        our_refresh: str | None = None
        if google_refresh:
            our_refresh = secrets.token_urlsafe(32)
            self._refresh_tokens[our_refresh] = RefreshToken(
                token=our_refresh,
                client_id=client_id,
                scopes=scopes,
            )
            self._refresh_google[our_refresh] = google_refresh

        return OAuthToken(
            access_token=our_access,
            token_type="Bearer",
            expires_in=max(1, int(expires_at - time.time())),
            scope=" ".join(scopes),
            refresh_token=our_refresh,
        )

    # ── Refresh ─────────────────────────────────────────────────────────────

    async def load_refresh_token(
        self, client: OAuthClientInformationFull, refresh_token: str
    ) -> RefreshToken | None:
        token = self._refresh_tokens.get(refresh_token)
        if token is None or token.client_id != client.client_id:
            return None
        return token

    async def exchange_refresh_token(
        self,
        client: OAuthClientInformationFull,
        refresh_token: RefreshToken,
        scopes: list[str],
    ) -> OAuthToken:
        google_refresh = self._refresh_google.get(refresh_token.token)
        if google_refresh is None:
            raise TokenError("invalid_grant", "unknown refresh token")

        try:
            async with httpx.AsyncClient() as http:
                resp = await http.post(
                    GOOGLE_TOKEN_URL,
                    data={
                        "refresh_token": google_refresh,
                        "client_id": self.google_client_id,
                        "client_secret": self.google_client_secret,
                        "grant_type": "refresh_token",
                    },
                )
        except Exception as exc:
            raise TokenError("server_error", "google refresh failed") from exc

        if resp.status_code != 200:
            raise TokenError("invalid_grant", "google refused the refresh token")

        google = resp.json()
        new_google_access = google.get("access_token", "")
        # Re-gate on refresh so revoked/expired accounts lose access promptly.
        if await self._verifier.verify_token(new_google_access) is None:
            raise TokenError("access_denied", "account no longer permitted")

        # Rotate: retire the old refresh token, issue a fresh pair. Google may or
        # may not return a new refresh token; keep the existing one if not.
        self._refresh_tokens.pop(refresh_token.token, None)
        self._refresh_google.pop(refresh_token.token, None)

        requested_scopes = scopes or refresh_token.scopes
        return self._issue_tokens(
            client_id=client.client_id,
            scopes=requested_scopes,
            resource=None,
            google_access=new_google_access,
            google_refresh=google.get("refresh_token") or google_refresh,
            expires_at=time.time() + int(google.get("expires_in", 3600)),
        )

    # ── Verification / revocation ───────────────────────────────────────────

    async def load_access_token(self, token: str) -> AccessToken | None:
        access = self._access_tokens.get(token)
        if access is None:
            return None
        if access.expires_at is not None and access.expires_at < time.time():
            self._access_tokens.pop(token, None)
            return None
        return access

    async def revoke_token(self, token: AccessToken | RefreshToken) -> None:
        # Drop whichever kind we were handed; best-effort, idempotent.
        self._access_tokens.pop(token.token, None)
        if self._refresh_tokens.pop(token.token, None) is not None:
            self._refresh_google.pop(token.token, None)
