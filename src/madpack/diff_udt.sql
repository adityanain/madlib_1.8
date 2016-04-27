SET client_min_messages to ERROR;

CREATE OR REPLACE FUNCTION filter_schema(argstr text, schema_name text)
RETURNS text AS $$
    if argstr is None:
        return "NULL"
    return argstr.replace(schema_name + ".", '')
$$ LANGUAGE plpythonu;

CREATE OR REPLACE FUNCTION get_types(schema_name text)
RETURNS VOID AS
$$
    import plpy
    plpy.execute("""
        CREATE TABLE types_{schema_name} AS
        SELECT n.nspname as "schema",
               filter_schema(pg_catalog.format_type(t.oid, NULL), '{schema_name}') AS "name",
               t.typrelid AS "typrelid"
        FROM pg_catalog.pg_type t
          LEFT JOIN pg_catalog.pg_namespace n
          ON n.oid = t.typnamespace
        WHERE (t.typrelid = 0 OR
               (SELECT c.relkind = 'c' FROM pg_catalog.pg_class c WHERE c.oid = t.typrelid))
          --AND t.typname !~ '^_'
          AND n.nspname ~ '^({schema_name})$'
        ORDER BY 1, 2;
        """.format(schema_name=schema_name))
$$ LANGUAGE plpythonu;


CREATE OR REPLACE FUNCTION detect_changed_types(
    common_udt_table  text
)
RETURNS TEXT[] AS
$$
    import plpy

    rv = plpy.execute("""
        SELECT name, old_relid, new_relid
        FROM {common_udt_table}
        """.format(common_udt_table=common_udt_table))
    changed_udt = []
    for r in rv:
        name = r['name']
        old_relid = r['old_relid']
        new_relid = r['new_relid']
        rv = plpy.execute("""
            SELECT
                array_eq(old_type, new_type) AS changed
            FROM
            (
                SELECT array_agg(a.attname || pg_catalog.format_type(a.atttypid, a.atttypmod) || a.attnum order by a.attnum) AS old_type
                FROM pg_catalog.pg_attribute a
                LEFT OUTER JOIN pg_catalog.pg_attribute_encoding e
                ON   e.attrelid = a .attrelid AND e.attnum = a.attnum
                WHERE a.attrelid = '{old_relid}' AND a.attnum > 0 AND NOT a.attisdropped
            ) t1,
            (
                SELECT array_agg(a.attname || pg_catalog.format_type(a.atttypid, a.atttypmod) || a.attnum order by a.attnum) AS new_type
                FROM pg_catalog.pg_attribute a
                LEFT OUTER JOIN pg_catalog.pg_attribute_encoding e
                ON   e.attrelid = a .attrelid AND e.attnum = a.attnum
                WHERE a.attrelid = '{new_relid}' AND a.attnum > 0 AND NOT a.attisdropped
            ) t2
            """.format(old_relid=old_relid, new_relid=new_relid))[0]['changed']
        if not rv:
            changed_udt.append(name)
    return changed_udt
$$ LANGUAGE plpythonu;

-- Get UDTs
DROP TABLE IF EXISTS types_madlib_v17;
DROP TABLE IF EXISTS types_madlib_v16;
SELECT get_types('madlib_v17');
SELECT get_types('madlib_v16');

--SELECT name FROM types_madlib;
--SELECT name FROM types_madlib_v15;

--Dropped
SELECT
    v16.name AS "Dropped UDTs"
FROM
    types_madlib_v16 AS v16
    LEFT JOIN
    types_madlib_v17 AS v17
    USING (name)
WHERE v17.name IS NULL;

--Added
-- SELECT
--     v17.name AS "Added UDTs"
-- FROM
--     types_madlib_v16 AS v16
--     RIGHT JOIN
--     types_madlib_v17 AS v17
--     USING (name)
-- WHERE v16.name IS NULL;

--Common
DROP TABLE IF EXISTS types_common;
CREATE TABLE types_common AS
SELECT
    v16.name, v16.typrelid AS old_relid, v17.typrelid AS new_relid
FROM
    types_madlib_v16 AS v16
    JOIN
    types_madlib_v17 AS v17
    USING (name)
WHERE v16.typrelid <> 0; -- 0 means base type

SELECT
    array_upper(detect_changed_types('types_common'), 1) AS N,
    detect_changed_types('types_common') AS "Changed UDTs";

