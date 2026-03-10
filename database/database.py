"""
database.py  —  MariaDB version
────────────────────────────────────────────────────────────────────────────
Drop-in replacement for the SQLite database.py.
All class / method signatures are identical — only the backend changes.

Connection config (env vars or constructor kwargs):
  DB_HOST       default: localhost
  DB_PORT       default: 3306
  DB_USER       default: root
  DB_PASSWORD   required
  DB_NAME       default: attendance_db
  DB_POOL_SIZE  default: 5
"""

from __future__ import annotations

import os
import struct
import logging
from datetime import datetime
from typing import Optional

import mysql.connector
from mysql.connector import pooling
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("database")


# ─────────────────────────────────────────────────────────────────────────────
# Status constants
# ─────────────────────────────────────────────────────────────────────────────

class AttendanceStatus:
    PRESENT = "P"
    ABSENT  = "A"
    LATE    = "L"
    LEAVE   = "LV"

    LABELS = {
        "P":  "Present",
        "A":  "Absent",
        "L":  "Late",
        "LV": "On Leave",
    }

    @classmethod
    def label(cls, code: str) -> str:
        return cls.LABELS.get(code, code or "Present")


# ─────────────────────────────────────────────────────────────────────────────
# DatabaseConfig
# ─────────────────────────────────────────────────────────────────────────────

class DatabaseConfig:
    """
    MariaDB connection pool manager.

    Can be configured via:
      1. Environment variables  (DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME)
      2. DatabaseConfig.configure(host=..., password=..., ...)  class method
      3. Both — kwargs override env vars
    """

    # Defaults — overridden by env vars or configure()
    _config: dict = {
        "host":              os.getenv("DB_HOST",      "localhost"),
        "port":        int(  os.getenv("DB_PORT",      "3306")),
        "user":              os.getenv("DB_USER",      "frs_user"),
        "password":          os.getenv("DB_PASSWORD",  ""),
        "database":          os.getenv("DB_NAME",      "attendance_db"),
        "pool_size":   int(  os.getenv("DB_POOL_SIZE", "5")),
        "pool_name":         "attendance_pool",
        "autocommit":        False,
        "charset":           "utf8mb4",
        "collation":         "utf8mb4_unicode_ci",
        "connection_timeout": 10,
    }

    _pool: Optional[pooling.MySQLConnectionPool] = None

    # ── configure ─────────────────────────────────────────────────────────────

    @classmethod
    def configure(
        cls,
        host:       str  = None,
        port:       int  = None,
        user:       str  = None,
        password:   str  = None,
        database:   str  = None,
        pool_size:  int  = None,
        charset:    str  = None,
        autocommit: bool = None,
        connection_timeout: int = None,
    ) -> None:
        """
        Override connection settings.py at runtime.

        Example
        ───────
        DatabaseConfig.configure(
            host="192.168.1.50",
            port=3306,
            user="app_user",
            password="s3cr3t",
            database="attendance_db",
            pool_size=10,
        )
        DatabaseConfig.initialize_tables()
        """
        if host       is not None: cls._config["host"]       = host
        if port       is not None: cls._config["port"]       = int(port)
        if user       is not None: cls._config["user"]       = user
        if password   is not None: cls._config["password"]   = password
        if database   is not None: cls._config["database"]   = database
        if pool_size  is not None: cls._config["pool_size"]  = int(pool_size)
        if charset    is not None: cls._config["charset"]    = charset
        if autocommit is not None: cls._config["autocommit"] = autocommit
        if connection_timeout is not None:
            cls._config["connection_timeout"] = connection_timeout

        # Reset pool so next get_connection() rebuilds it
        cls._pool = None
        logger.info(
            "DatabaseConfig updated → %s@%s:%s/%s (pool=%s)",
            cls._config["user"], cls._config["host"],
            cls._config["port"], cls._config["database"],
            cls._config["pool_size"],
        )

    # ── pool / connection ──────────────────────────────────────────────────────

    @classmethod
    def _get_pool(cls) -> pooling.MySQLConnectionPool:
        if cls._pool is None:
            cls._pool = pooling.MySQLConnectionPool(
                pool_name      = cls._config["pool_name"],
                pool_size      = cls._config["pool_size"],
                host           = cls._config["host"],
                port           = cls._config["port"],
                user           = cls._config["user"],
                password       = cls._config["password"],
                database       = cls._config["database"],
                charset        = cls._config["charset"],
                collation      = cls._config["collation"],
                autocommit     = cls._config["autocommit"],
                connection_timeout = cls._config["connection_timeout"],
            )
            logger.info(
                "MariaDB pool created → %s@%s:%s/%s  (pool_size=%d)",
                cls._config["user"], cls._config["host"],
                cls._config["port"], cls._config["database"],
                cls._config["pool_size"],
            )
        return cls._pool

    @classmethod
    def get_connection(cls):
        """Return a pooled MariaDB connection, or None on failure."""
        try:
            conn = cls._get_pool().get_connection()
            # ✅ Ping: always fetch and close so the result buffer is cleared
            ping_cursor = conn.cursor()
            ping_cursor.execute("SELECT 1")
            ping_cursor.fetchall()  # consume the result
            ping_cursor.close()  # close the cursor
            return conn
        except Exception as exc:
            logger.error("DB connection failed: %s", exc)
            return None

    # ── schema ────────────────────────────────────────────────────────────────

    @classmethod
    def initialize_tables(cls) -> None:
        """Create all tables (idempotent — safe to call on every startup)."""
        conn = cls.get_connection()
        if not conn:
            logger.error("Cannot initialise tables — no DB connection.")
            return

        cursor = conn.cursor()
        try:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id     INT          AUTO_INCREMENT PRIMARY KEY,
                    name        VARCHAR(128) NOT NULL,
                    email       VARCHAR(128),
                    phone       VARCHAR(32),
                    department  VARCHAR(64),
                    role        VARCHAR(32)  DEFAULT 'Employee',
                    created_at  DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    updated_at  DATETIME     DEFAULT CURRENT_TIMESTAMP
                                ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_name ON users(name)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_role ON users(role)")

            # Attendance
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id               INT          AUTO_INCREMENT PRIMARY KEY,
                    user_id          INT          NOT NULL,
                    date             DATE         NOT NULL,
                    time             TIME         NOT NULL,
                    status           CHAR(2)      DEFAULT 'P',
                    image_path       TEXT,
                    confidence_score FLOAT,
                    camera_id        VARCHAR(64),
                    created_at       DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE KEY uq_user_date (user_id, date),
                    CONSTRAINT fk_att_user FOREIGN KEY (user_id)
                        REFERENCES users(user_id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_date     ON attendance(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_user_date ON attendance(user_id, date)")

            # Cameras
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS cameras (
                    camera_id     VARCHAR(64)  PRIMARY KEY,
                    camera_name   VARCHAR(128) NOT NULL,
                    camera_source VARCHAR(256) NOT NULL,
                    status        VARCHAR(16)  DEFAULT 'active',
                    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)
            # watchlist

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist (
                    watchlist_id  VARCHAR(64)  PRIMARY KEY,
                    user_id       INT          NOT NULL,
                    category      ENUM('blacklist','whitelist','vip') DEFAULT 'blacklist',
                    alert_enabled TINYINT(1)   DEFAULT 1,
                    alarm_enabled TINYINT(1)   DEFAULT 1,
                    threshold     FLOAT        DEFAULT 0.75,
                    cooldown_sec  INT          DEFAULT 10,
                    active        TINYINT(1)   DEFAULT 1,
                    created_at    DATETIME     DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT fk_wl_user FOREIGN KEY (user_id)
                        REFERENCES users(user_id) ON DELETE CASCADE
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            cursor.execute("CREATE INDEX IF NOT EXISTS idx_watchlist_user ON watchlist(user_id)")

            # Watchlist events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS watchlist_events (
                    id               INT         AUTO_INCREMENT PRIMARY KEY,
                    watchlist_id     VARCHAR(64),
                    user_id          VARCHAR(64),
                    camera_id        VARCHAR(64),
                    confidence_score FLOAT,
                    image_path       TEXT,
                    alarm_triggered  TINYINT(1),
                    created_at       DATETIME    DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
            """)

            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_we_date ON watchlist_events(created_at)"
            )

            conn.commit()
            logger.info("MariaDB tables initialised successfully.")

        except Exception as exc:
            conn.rollback()
            logger.error("initialize_tables failed: %s", exc)
            raise
        finally:
            cursor.close()
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# AttendanceManager
# ─────────────────────────────────────────────────────────────────────────────

class AttendanceManager:

    def __init__(self) -> None:
        self.today_attendance_cache: set[str] = set()
        self._load_today_cache()

    def _load_today_cache(self) -> None:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return
        try:
            today  = datetime.now().date().isoformat()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT user_id FROM attendance WHERE date = %s", (today,))
            self.today_attendance_cache = {r["user_id"] for r in cursor.fetchall()}
        except Exception as exc:
            logger.error("Loading attendance cache: %s", exc)
        finally:
            cursor.close()
            conn.close()

    def mark_attendance(
        self,
        user_id:          str,
        confidence_score: float,
        image_path:       str  = None,
        camera_id:        str  = None,
        status:           str  = AttendanceStatus.PRESENT,
    ) -> bool:
        if user_id in self.today_attendance_cache:
            return False

        conn = DatabaseConfig.get_connection()
        if not conn:
            return False

        try:
            cursor = conn.cursor(dictionary=True)
            today        = datetime.now().date().isoformat()
            current_time = datetime.now().strftime("%H:%M:%S")

            # Auto-create user if missing
            cursor.execute("SELECT user_id FROM users WHERE user_id = %s", (user_id,))
            if not cursor.fetchone():
                cursor.execute(
                    "INSERT INTO users (user_id, name, department, role) VALUES (%s,%s,%s,%s)",
                    (user_id, f"User {user_id[-8:]}", "Unknown", "Employee"),
                )
                logger.info("Auto-created user %s", user_id)

            # Normalise confidence_score
            if isinstance(confidence_score, bytes):
                confidence_score = struct.unpack("f", confidence_score)[0]
            else:
                confidence_score = float(confidence_score)

            cursor.execute(
                """
                INSERT INTO attendance
                    (user_id, date, time, status, image_path, confidence_score, camera_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (user_id, today, current_time, status,
                 image_path, confidence_score, camera_id),
            )
            conn.commit()
            self.today_attendance_cache.add(user_id)
            logger.info(
                "Attendance marked: %s [%s] %s",
                user_id, status, AttendanceStatus.label(status),
            )
            return True

        except Exception as exc:
            conn.rollback()
            if "Duplicate entry" in str(exc):
                self.today_attendance_cache.add(user_id)
                return False
            logger.error("mark_attendance error: %s", exc)
            return False
        finally:
            cursor.close()
            conn.close()

    def get_attendance_records(
        self,
        start_date: str  = None,
        end_date:   str  = None,
        user_id:    str  = None,
    ) -> list[dict]:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return []

        try:
            cursor = conn.cursor(dictionary=True)
            query  = """
                SELECT
                    a.id, a.user_id, a.date, a.time, a.status,
                    a.image_path, a.confidence_score, a.camera_id, a.created_at,
                    u.name, u.department, u.role, u.email, u.phone
                FROM attendance a
                JOIN users u ON a.user_id = u.user_id
                WHERE 1=1
            """
            params = []

            if start_date:
                query += " AND a.date >= %s"; params.append(start_date)
            if end_date:
                query += " AND a.date <= %s"; params.append(end_date)
            if user_id:
                query += " AND a.user_id = %s"; params.append(user_id)

            query += " ORDER BY a.date DESC, a.time DESC"
            cursor.execute(query, params)
            rows = cursor.fetchall()

            records = []
            for row in rows:
                conf = row.get("confidence_score")
                if isinstance(conf, bytes):
                    try:    row["confidence_score"] = struct.unpack("f", conf)[0]
                    except: row["confidence_score"] = 0.0
                elif conf is None:
                    row["confidence_score"] = 0.0
                else:
                    row["confidence_score"] = float(conf)

                if not row.get("status"):
                    row["status"] = AttendanceStatus.PRESENT

                # Convert date/time objects to strings for JSON serialisation
                if hasattr(row.get("date"), "isoformat"):
                    row["date"] = row["date"].isoformat()
                if hasattr(row.get("time"), "total_seconds"):
                    row["time"] = str(row["time"])

                records.append(row)

            logger.debug("Retrieved %d attendance records", len(records))
            return records

        except Exception as exc:
            logger.error("get_attendance_records: %s", exc)
            return []
        finally:
            cursor.close()
            conn.close()

    def get_all_attendance_details(self) -> list[dict]:
        return self.get_attendance_records()


# ─────────────────────────────────────────────────────────────────────────────
# UserManager
# ─────────────────────────────────────────────────────────────────────────────

class UserManager:

    @staticmethod
    def add_user(
        user_id:    str,
        name:       str,
        email:      str = None,
        phone:      str = None,
        department: str = None,
        role:       str = "Employee",
    ) -> bool:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return False
        try:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO users (user_id, name, email, phone, department, role)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (user_id, name, email, phone, department, role),
            )
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            if "Duplicate entry" in str(exc):
                logger.warning("User %s already exists", user_id)
            else:
                logger.error("add_user: %s", exc)
            return False
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def get_user(user_id: int) -> Optional[dict]:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return None
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE user_id = %s", (user_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
        except Exception as exc:
            logger.error("get_user: %s", exc)
            return None
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def add_user_auto(
            name: str,
            email: str = None,
            phone: str = None,
            department: str = None,
            role: str = "Employee",
    ) -> int | None:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return None
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO users (name, email, phone, department, role) VALUES (%s,%s,%s,%s,%s)",
                (name, email, phone, department, role),
            )
            conn.commit()
            return cursor.lastrowid  # ✅ auto-generated INT id
        except Exception as exc:
            conn.rollback()
            logger.error("add_user_auto: %s", exc)
            return None
        finally:
            if cursor: cursor.close()
            conn.close()

    @staticmethod
    def delete_user(user_id: int) -> bool:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return False
        cursor = None
        try:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE user_id = %s", (user_id,))
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.error("delete_user: %s", exc)
            return False
        finally:
            if cursor: cursor.close()
            conn.close()

    @staticmethod
    def get_user_by_id(user_id: str) -> Optional[dict]:
        return UserManager.get_user(user_id)

    @staticmethod
    def get_all_users() -> list[dict]:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return []
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users ORDER BY name")
            return [dict(r) for r in cursor.fetchall()]
        except Exception as exc:
            logger.error("get_all_users: %s", exc)
            return []
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def update_user(user_id: int, **kwargs) -> bool:
        conn = DatabaseConfig.get_connection()
        if not conn:
            return False

        fields = [f"{k} = %s" for k, v in kwargs.items() if v is not None]
        values = [v for v in kwargs.values() if v is not None]
        if not fields:
            return False

        values.append(user_id)
        try:
            cursor = conn.cursor()
            cursor.execute(
                f"UPDATE users SET {', '.join(fields)} WHERE user_id = %s", values
            )
            conn.commit()
            return True
        except Exception as exc:
            conn.rollback()
            logger.error("update_user: %s", exc)
            return False
        finally:
            cursor.close()
            conn.close()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_database(
    host:      str = None,
    port:      int = None,
    user:      str = None,
    password:  str = None,
    database:  str = None,
    pool_size: int = None,
) -> None:
    """
    One-shot initialiser — configure + create tables.

    Usage
    ─────
    # purely via env vars:
    init_database()

    # or pass everything explicitly:
    init_database(
        host="192.168.1.50",
        port=3306,
        user="app_user",
        password="s3cr3t",
        database="attendance_db",
        pool_size=10,
    )
    """
    if any(v is not None for v in (host, port, user, password, database, pool_size)):
        DatabaseConfig.configure(
            host=host, port=port, user=user,
            password=password, database=database,
            pool_size=pool_size,
        )
    DatabaseConfig.initialize_tables()
    logger.info("MariaDB initialisation complete.")