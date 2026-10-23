from app.services.schema_compiler import compile_table_request, resolve_dialect
from app.services.validator import validate_sql


def test_plain_table_request_defaults_to_postgresql_and_uses_safe_template():
    requirement = "Create a table employee"
    dialect, source = resolve_dialect(None, requirement)
    result = compile_table_request(requirement, dialect)

    assert dialect == "postgresql"
    assert source == "default"
    assert result is not None
    assert 'CREATE TABLE "employee"' in result.output
    assert "employee_id" in result.output
    assert validate_sql(result.output, dialect).validation_status == "valid"


def test_explicit_columns_and_mysql_are_rendered_correctly():
    requirement = (
        "Create a table named employees with employee_id integer primary key, "
        "full_name text, email varchar(255) unique in mysql"
    )
    dialect, source = resolve_dialect(None, requirement)
    result = compile_table_request(requirement, dialect)

    assert dialect == "mysql"
    assert source == "requirement"
    assert result is not None
    assert "CREATE TABLE `employees`" in result.output
    assert "`employee_id` INTEGER AUTO_INCREMENT PRIMARY KEY" in result.output
    assert "`email` VARCHAR(255) UNIQUE" in result.output
    assert validate_sql(result.output, dialect).validation_status == "valid"


def test_dbml_request_is_supported_and_validated():
    requirement = "Create a table named events with id, title, and created_at in DBML"
    dialect, _ = resolve_dialect(None, requirement)
    result = compile_table_request(requirement, dialect)

    assert dialect == "dbml"
    assert result is not None
    assert result.output.startswith("Table events {")
    validation = validate_sql(result.output, dialect)
    assert validation.valid_syntax is True
    assert validation.referenced_objects == ["events"]

