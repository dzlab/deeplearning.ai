-- 000_c_common_user.sql
-- Run as SYSTEM against the CDB ROOT (service FREE / CDB$ROOT).
--
-- Fallback for sandboxes that expose ONLY the CDB root with no usable
-- application PDB (v$pdbs shows just PDB$SEED) and only `system`-over-network
-- access (no SYSDBA, so a freshly CREATEd PDB cannot be OPENed and is unusable).
-- In that case the course runs as a COMMON user in the root instead of dmuser
-- in a PDB. The setup script substitutes the name/password before applying:
--   sed -e 's/&&CUSER/C##DLAI/g' -e 's/&&CPWD/<pwd>/g' 000_c_common_user.sql
-- (the &&-vars are bound by the Python applier, which does the substitution).
--
-- Why these grants:
--   * Common user name MUST start with C## (ORA-65096 otherwise).
--   * DEFAULT TABLESPACE USERS: USERS uses ASSM; oracleagentmemory creates
--     JSON-typed columns, which ORA-43853 forbids in a non-ASSM tablespace
--     (SYSTEM is non-ASSM). USERS exists in the root too.
--   * UNLIMITED TABLESPACE: DBMS_VECTOR.LOAD_ONNX_MODEL writes the ~90 MB model
--     to the user's default tablespace; without quota it dies with ORA-01950
--     (and, as plain system, the session can drop with DPY-4011 instead).
--   * DB_DEVELOPER_ROLE: HNSW / VECTOR DDL on the container-registry image.
--
-- Idempotent: re-creating an existing user (ORA-01920) or re-granting without
-- privilege (ORA-01031) is swallowed; tablespace/role are (re)applied each run.

BEGIN
  EXECUTE IMMEDIATE
    'CREATE USER &&CUSER IDENTIFIED BY "&&CPWD" DEFAULT TABLESPACE USERS CONTAINER=ALL';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE NOT IN (-1920, -1031) THEN RAISE; END IF;
END;
/

GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW, CREATE PROCEDURE TO &&CUSER CONTAINER=ALL;
/

ALTER USER &&CUSER DEFAULT TABLESPACE USERS;
/

-- UNLIMITED TABLESPACE covers the model load regardless of default tablespace.
GRANT UNLIMITED TABLESPACE TO &&CUSER CONTAINER=ALL;
/

BEGIN
  EXECUTE IMMEDIATE 'GRANT DB_DEVELOPER_ROLE TO &&CUSER CONTAINER=ALL';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE NOT IN (-1031, -1720) THEN RAISE; END IF;
END;
/
