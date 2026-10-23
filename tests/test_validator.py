from app.services.validator import validate_sql


def test_valid_create_table_parses():
    sql = (
        "CREATE TABLE customers ("
        "customer_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY, "
        "email VARCHAR(255) NOT NULL UNIQUE);"
    )
    result = validate_sql(sql)
    assert result.valid_syntax is True
    assert result.sql.startswith("CREATE TABLE")


def test_invalid_sql_is_rejected():
    result = validate_sql("CREATE TABLE (")
    assert result.valid_syntax is False
    assert result.warnings


def test_drop_table_is_flagged_as_destructive():
    result = validate_sql("DROP TABLE customers;")
    assert result.valid_syntax is True
    assert any("Destructive" in warning for warning in result.warnings)


def test_missing_primary_key_adds_warning():
    result = validate_sql("CREATE TABLE notes (body TEXT);")
    assert result.valid_syntax is True
    assert any("primary key" in warning.lower() for warning in result.warnings)
