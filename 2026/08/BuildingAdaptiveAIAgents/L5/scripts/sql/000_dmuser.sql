-- 000_dmuser.sql
-- Run as SYSTEM against FREEPDB1.
-- Creates the dmuser account used by oracleagentmemory and the course scripts.
--
-- Password is hardcoded to "continual_learning" — suitable for the Oracle Free
-- container used in this course.  To use a different password, substitute the
-- string before piping to sqlplus:
--   sed 's/continual_learning/<your_password>/g' 000_dmuser.sql | sqlplus -s system/...
--
-- The idempotency guard catches ORA-01920 (user already exists) and
-- ORA-01031 (insufficient privileges when re-granting) and continues silently.
--
-- TABLESPACE: dmuser must default to an ASSM tablespace because
-- oracleagentmemory creates JSON-typed columns, which ORA-43853 forbids in a
-- non-ASSM tablespace (SYSTEM is non-ASSM).  We do NOT hardcode "USERS": some
-- Oracle Free / hosted PDB images ship without it (ORA-00959 'tablespace USERS
-- does not exist'), which previously aborted CREATE USER and forced the course
-- to fall back to running as SYSTEM.  Instead we resolve the PDB's default
-- permanent tablespace (ASSM by construction on Free), creating USERS only as a
-- last-resort fallback if no usable default is found.

-- Ensure a usable ASSM permanent tablespace exists; if the PDB's default
-- permanent tablespace is missing or is SYSTEM (non-ASSM), create USERS.
DECLARE
  v_ts    VARCHAR2(128);
  v_assm  VARCHAR2(30);
BEGIN
  BEGIN
    SELECT property_value INTO v_ts
      FROM database_properties
     WHERE property_name = 'DEFAULT_PERMANENT_TABLESPACE';
  EXCEPTION
    WHEN NO_DATA_FOUND THEN v_ts := NULL;
  END;

  -- Validate the resolved default: it must exist AND use automatic segment
  -- space management (ASSM).  SYSTEM/SYSAUX are MANUAL and would hit ORA-43853.
  IF v_ts IS NOT NULL THEN
    BEGIN
      SELECT segment_space_management INTO v_assm
        FROM dba_tablespaces WHERE tablespace_name = v_ts;
    EXCEPTION
      WHEN NO_DATA_FOUND THEN v_assm := NULL;
    END;
    IF v_assm IS NULL OR v_assm <> 'AUTO' THEN
      v_ts := NULL;  -- unusable; fall through to create USERS
    END IF;
  END IF;

  -- No usable ASSM default: create USERS (ASSM) and make it the PDB default.
  IF v_ts IS NULL THEN
    DECLARE
      v_exists NUMBER;
      v_omf    VARCHAR2(4000);
      v_seed   VARCHAR2(4000);
      v_dir    VARCHAR2(4000);
      v_df     VARCHAR2(4000);
    BEGIN
      SELECT COUNT(*) INTO v_exists
        FROM dba_tablespaces WHERE tablespace_name = 'USERS';
      IF v_exists = 0 THEN
        -- Prefer Oracle-Managed Files: with db_create_file_dest set, a DATAFILE
        -- clause needs no filename.  Many Oracle Free / hosted PDBs DON'T set
        -- it, and a bare 'DATAFILE SIZE ...' then fails ORA-02236 ('invalid file
        -- name') — the very failure that forced the fallback to SYSTEM.  When
        -- OMF is absent, derive an explicit path from an existing datafile's
        -- directory (e.g. .../FREEPDB1/system01.dbf -> .../FREEPDB1/users01.dbf).
        BEGIN
          SELECT value INTO v_omf
            FROM v$parameter WHERE name = 'db_create_file_dest';
        EXCEPTION
          WHEN NO_DATA_FOUND THEN v_omf := NULL;
        END;

        IF v_omf IS NOT NULL THEN
          EXECUTE IMMEDIATE
            'CREATE TABLESPACE USERS DATAFILE SIZE 200M AUTOEXTEND ON '||
            'NEXT 50M MAXSIZE 8G SEGMENT SPACE MANAGEMENT AUTO';
        ELSE
          SELECT file_name INTO v_seed
            FROM dba_data_files
           WHERE tablespace_name = 'SYSTEM' AND ROWNUM = 1;
          -- Strip the trailing filename, keep the directory (handles / and \).
          v_dir := SUBSTR(v_seed, 1,
                          GREATEST(INSTR(v_seed, '/', -1),
                                   INSTR(v_seed, '\', -1)));
          v_df  := v_dir || 'users01.dbf';
          EXECUTE IMMEDIATE
            'CREATE TABLESPACE USERS DATAFILE '''||v_df||''' SIZE 200M '||
            'AUTOEXTEND ON NEXT 50M MAXSIZE 8G SEGMENT SPACE MANAGEMENT AUTO';
        END IF;
      END IF;
    END;
    EXECUTE IMMEDIATE
      'ALTER DATABASE DEFAULT TABLESPACE USERS';
  END IF;
END;
/

-- Create dmuser defaulting to the (now guaranteed) ASSM default tablespace.
DECLARE
  v_ts VARCHAR2(128);
BEGIN
  SELECT property_value INTO v_ts
    FROM database_properties
   WHERE property_name = 'DEFAULT_PERMANENT_TABLESPACE';
  EXECUTE IMMEDIATE
    'CREATE USER dmuser IDENTIFIED BY "continual_learning" '||
    'DEFAULT TABLESPACE '||DBMS_ASSERT.ENQUOTE_NAME(v_ts);
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE NOT IN (-1920, -1031) THEN RAISE; END IF;
END;
/

GRANT CREATE SESSION, CREATE TABLE, CREATE SEQUENCE, CREATE VIEW, CREATE PROCEDURE TO dmuser;
/

-- Re-assert dmuser's default tablespace + quota against the PDB default ASSM
-- tablespace.  Applied unconditionally so a pre-existing dmuser (possibly left
-- pointing at SYSTEM by an earlier failed bootstrap) is corrected on re-run.
DECLARE
  v_ts VARCHAR2(128);
BEGIN
  SELECT property_value INTO v_ts
    FROM database_properties
   WHERE property_name = 'DEFAULT_PERMANENT_TABLESPACE';
  EXECUTE IMMEDIATE
    'ALTER USER dmuser DEFAULT TABLESPACE '||DBMS_ASSERT.ENQUOTE_NAME(v_ts);
  EXECUTE IMMEDIATE
    'ALTER USER dmuser QUOTA UNLIMITED ON '||DBMS_ASSERT.ENQUOTE_NAME(v_ts);
END;
/

-- Grant DB_DEVELOPER_ROLE for HNSW / VECTOR DDL on container-registry image.
BEGIN
  EXECUTE IMMEDIATE 'GRANT DB_DEVELOPER_ROLE TO dmuser';
EXCEPTION
  WHEN OTHERS THEN
    IF SQLCODE NOT IN (-1031, -1720) THEN RAISE; END IF;
END;
/
