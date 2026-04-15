"""Static dictionary for multi-query embedding expansion (Story 8.5).

Maps Portuguese/abstract terms to English code terminology.
Add entries here to improve retrieval without code changes.

Format: each key is a substring to match (lowercase, after stop-word removal).
Values are lists of English code terms to include in the reformulation vector.
"""

from __future__ import annotations

# Module-level constant — extensible without code changes
EXPANSION_DICT: dict[str, list[str]] = {
    # Feature access / gating
    "feature": ["feature", "access", "gate", "permission", "capability"],
    "funcionalidade": ["feature", "access", "gate", "permission", "capability"],
    # Plan / subscription / pricing
    "plano": ["plan", "subscription", "tier", "pricing"],
    "planos": ["plan", "subscription", "tier", "pricing"],
    "assinatura": ["subscription", "plan", "tier", "billing"],
    "assinaturas": ["subscription", "plan", "tier", "billing"],
    "preco": ["price", "pricing", "cost", "billing"],
    # Limits / restrictions / quotas
    "limita": ["limit", "restrict", "guard", "check", "validate", "cap", "quota"],
    "limite": ["limit", "restrict", "guard", "check", "validate", "cap", "quota"],
    "limites": ["limit", "restrict", "guard", "check", "validate", "cap", "quota"],
    "limitacao": ["limit", "restrict", "quota", "cap", "threshold"],
    "cota": ["quota", "limit", "cap", "allowance"],
    "threshold": ["threshold", "limit", "cap", "boundary"],
    # Usage / consumption
    "uso": ["usage", "access", "consumption", "utilization"],
    "utilizacao": ["usage", "utilization", "consumption"],
    "consumo": ["consumption", "usage", "utilization"],
    # Authentication / authorization
    "autenticacao": ["auth", "authentication", "token", "session", "jwt"],
    "autenticar": ["auth", "authenticate", "login", "token", "session"],
    "autorizacao": ["authorization", "permission", "access", "role", "scope"],
    "autorizar": ["authorize", "permission", "access", "role"],
    "login": ["login", "auth", "authenticate", "session", "credential"],
    "senha": ["password", "credential", "hash", "bcrypt"],
    "token": ["token", "jwt", "session", "bearer", "access_token"],
    # Errors / exceptions / failures
    "erro": ["error", "exception", "failure", "catch", "throw"],
    "erros": ["error", "exception", "failure", "catch", "throw"],
    "excecao": ["exception", "error", "catch", "throw", "raise"],
    "falha": ["failure", "error", "exception", "fault"],
    "falhar": ["fail", "error", "exception", "fault"],
    # User / account
    "usuario": ["user", "account", "member", "profile"],
    "usuarios": ["user", "account", "member", "profile"],
    "conta": ["account", "user", "profile", "tenant"],
    "contas": ["account", "user", "profile", "tenant"],
    "perfil": ["profile", "user", "account", "settings"],
    # Payment / billing / financial
    "pagamento": ["payment", "billing", "invoice", "charge", "transaction"],
    "pagamentos": ["payment", "billing", "invoice", "charge", "transaction"],
    "cobranca": ["billing", "charge", "invoice", "payment"],
    "fatura": ["invoice", "billing", "payment", "charge"],
    "cartao": ["card", "credit_card", "stripe", "payment"],
    # Notification / email / messaging
    "notificacao": ["notification", "notify", "alert", "message", "event"],
    "notificacoes": ["notification", "notify", "alert", "message", "event"],
    "email": ["email", "mail", "smtp", "message", "notification"],
    "mensagem": ["message", "notification", "event", "payload"],
    # Configuration / settings
    "configuracao": ["config", "configuration", "settings", "options", "params"],
    "configuracoes": ["config", "configuration", "settings", "options"],
    "config": ["config", "configuration", "settings", "options"],
    # Database / persistence / storage
    "banco": ["database", "db", "storage", "repository", "persistence"],
    "dados": ["data", "model", "schema", "record", "entity"],
    "repositorio": ["repository", "repo", "storage", "dao", "service"],
    "salvar": ["save", "persist", "store", "write", "upsert"],
    "buscar": ["fetch", "query", "find", "search", "retrieve", "get"],
    # API / endpoints / routes
    "rota": ["route", "endpoint", "path", "handler", "controller"],
    "rotas": ["route", "endpoint", "path", "handler", "controller"],
    "endpoint": ["endpoint", "route", "handler", "controller", "api"],
    "requisicao": ["request", "http", "api", "endpoint", "handler"],
    # Caching
    "cache": ["cache", "memoize", "redis", "ttl", "invalidate"],
    "cachear": ["cache", "memoize", "store", "ttl"],
    # Events / webhooks
    "evento": ["event", "webhook", "trigger", "dispatch", "emit"],
    "eventos": ["event", "webhook", "trigger", "dispatch", "emit"],
    # Validation
    "validacao": ["validation", "validate", "schema", "constraint", "check"],
    "validar": ["validate", "check", "verify", "constraint", "schema"],
    # Logging / monitoring
    "log": ["log", "logging", "monitor", "trace", "audit"],
    "logs": ["log", "logging", "monitor", "trace", "audit"],
    "monitorar": ["monitor", "observe", "track", "log", "metric"],
    # Tests
    "teste": ["test", "spec", "mock", "assert", "fixture"],
    "testes": ["test", "spec", "mock", "assert", "fixture"],
    # Background jobs / async
    "tarefa": ["task", "job", "worker", "background", "queue"],
    "fila": ["queue", "worker", "background", "job", "async"],
}
