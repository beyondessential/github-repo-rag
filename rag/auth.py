"""
Google OAuth token verification for the MCP server (HTTP transport).

Implements MCP's TokenVerifier protocol — validates Google OAuth2 access tokens
via Google's tokeninfo endpoint and enforces domain/email restrictions.
"""

import httpx
from mcp.server.auth.provider import AccessToken

GOOGLE_TOKENINFO_URL = "https://www.googleapis.com/oauth2/v3/tokeninfo"


class GoogleTokenVerifier:
    """Validates Google OAuth2 access tokens for MCP HTTP transport.

    Tokens are verified against Google's tokeninfo API. Callers must have
    obtained a Google access token with at least the ``openid email`` scopes
    (e.g. via ``gcloud auth print-access-token`` or any standard OAuth flow).

    Access is restricted by domain and/or email allowlist:
        - ``allowed_domain``: only emails from this domain are accepted
          (e.g. ``"beyondessential.com.au"``)
        - ``allowed_emails``: only these specific emails are accepted
        - If both are empty, any verified Google account is accepted.
    """

    def __init__(
        self,
        allowed_domain: str = "",
        allowed_emails: list[str] | None = None,
    ) -> None:
        self.allowed_domain = allowed_domain
        self.allowed_emails = set(allowed_emails or [])

    async def verify_token(self, token: str) -> AccessToken | None:
        """Validate a Google OAuth2 access token.

        Returns an AccessToken if valid and permitted, None otherwise.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(
                    GOOGLE_TOKENINFO_URL,
                    params={"access_token": token},
                )

            if resp.status_code != 200:
                return None

            info = resp.json()
            if "error" in info:
                return None

            email: str = info.get("email", "")
            if not email:
                return None  # token missing email scope
            if not info.get("email_verified"):
                return None  # unverified email

            # Domain restriction
            if self.allowed_domain and not email.endswith(f"@{self.allowed_domain}"):
                return None

            # Email allowlist
            if self.allowed_emails and email not in self.allowed_emails:
                return None

            return AccessToken(
                token=token,
                client_id=email,
                scopes=info.get("scope", "").split() if info.get("scope") else [],
                expires_at=int(info["exp"]) if info.get("exp") else None,
            )

        except Exception:
            return None
