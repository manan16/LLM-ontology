from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from app.config import Settings
from app.logger import get_logger


logger = get_logger(__name__)


class Neo4jConnectionError(RuntimeError):
    pass


class Neo4jClient:
    def __init__(self, settings: Settings) -> None:
        self.uri = settings.neo4j_uri
        self.username = settings.neo4j_username
        self.database = settings.neo4j_database
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_username, settings.neo4j_password),
        )
        self.verify_connectivity()

    def close(self) -> None:
        self.driver.close()

    def verify_connectivity(self) -> None:
        try:
            logger.debug("Verifying Neo4j connectivity uri=%s database=%s", self.uri, self.database)
            self.driver.verify_connectivity()
        except ServiceUnavailable as exc:
            raise Neo4jConnectionError(
                f"Could not connect to Neo4j at {self.uri}. "
                "If you want a local database, start Neo4j so it listens on this Bolt address. "
                "If you want Neo4j Aura, update NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD, and NEO4J_DATABASE in .env."
            ) from exc
        except AuthError as exc:
            raise Neo4jConnectionError(
                f"Connected to Neo4j at {self.uri}, but authentication failed for user {self.username}. "
                "Check NEO4J_USERNAME and NEO4J_PASSWORD in .env."
            ) from exc
        except ConfigurationError as exc:
            raise Neo4jConnectionError(
                f"Neo4j configuration is invalid for URI {self.uri}. Check NEO4J_URI in .env."
            ) from exc

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        retry=retry_if_exception_type(Exception),
    )
    def run_query(self, query: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        logger.debug(
            "Running Neo4j query database=%s query_head=%s param_keys=%s",
            self.database,
            " ".join(query.strip().split())[:160],
            sorted((parameters or {}).keys()),
        )
        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters or {})
            records = [record.data() for record in result]
            logger.debug("Neo4j query completed database=%s records=%s", self.database, len(records))
            return records

    def run_statements(self, statements: Iterable[str]) -> None:
        with self.driver.session(database=self.database) as session:
            for statement in statements:
                logger.debug("Running schema statement: %s", statement)
                session.run(statement)
