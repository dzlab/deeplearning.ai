-- 005_memory_graph.sql
-- Course-owned MEMORY_GRAPH_NODES + MEMORY_GRAPH_EDGES tables.
-- Run as dmuser against FREEPDB1.
--
-- Persists the structure-aware-retrieval graph topology so Module 3's
-- autonomous "graphify every 24h" loop is a real DB write -> reload -> re-rank
-- cycle (non-parametric continual learning: add rows, never retrain).
--
-- Idempotency: ORA-00955 (name already used) is silently swallowed.

BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE MEMORY_GRAPH_NODES (
      id         VARCHAR2(256) NOT NULL,
      domain     VARCHAR2(32)  NOT NULL,
      text       CLOB,
      updated_ts TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
      CONSTRAINT pk_memory_graph_nodes PRIMARY KEY (id, domain)
    )
  ';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN RAISE; END IF;
END;
/

BEGIN
  EXECUTE IMMEDIATE '
    CREATE TABLE MEMORY_GRAPH_EDGES (
      src        VARCHAR2(256) NOT NULL,
      dst        VARCHAR2(256) NOT NULL,
      kind       VARCHAR2(32)  DEFAULT ''import'',
      weight     NUMBER        DEFAULT 1.0,
      domain     VARCHAR2(32)  NOT NULL,
      updated_ts TIMESTAMP     DEFAULT CURRENT_TIMESTAMP,
      CONSTRAINT pk_memory_graph_edges PRIMARY KEY (src, dst, kind, domain)
    )
  ';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE != -955 THEN RAISE; END IF;
END;
/
