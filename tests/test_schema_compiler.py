from app.services.schema_compiler import compile_ddl_request, compile_table_request, resolve_dialect
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


def test_create_table_without_an_article_uses_the_real_table_name():
    result = compile_table_request("create table employee", "postgresql")

    assert result is not None
    assert result.output.startswith('CREATE TABLE "employee"')
    assert 'CREATE TABLE "create"' not in result.output


def test_common_action_phrases_and_dialect_suffixes_do_not_use_the_verb_as_name():
    cases = [
        ("please create table employee in postgres", "employee", "postgresql"),
        ("make table customers in mysql", "customers", "mysql"),
        ("build table products in sqlite", "products", "sqlite"),
        ("create table users in dbml", "users", "dbml"),
    ]

    for requirement, table_name, dialect in cases:
        result = compile_table_request(requirement, dialect)
        assert result is not None
        assert table_name in result.table_names
        assert "create" not in result.table_names
        assert result.dialect == dialect


def test_column_suffix_is_not_included_in_the_column_name():
    result = compile_table_request(
        "Create table employee with employee_id, email column, and phone column",
        "postgresql",
    )

    assert result is not None
    assert '"email" VARCHAR(255)' in result.output
    assert '"phone" TEXT' in result.output
    assert "email_column" not in result.output
    assert "phone_column" not in result.output


def test_foreign_key_column_is_not_silently_promoted_to_primary_key():
    result = compile_table_request(
        "Create table orders with customer_id integer references customers(id) in mysql",
        "mysql",
    )

    assert result is not None
    customer_line = next(line for line in result.output.splitlines() if "`customer_id`" in line)
    assert "`customer_id` INTEGER REFERENCES `customers` (`id`)" in customer_line
    assert "AUTO_INCREMENT" not in customer_line
    assert "PRIMARY KEY" not in customer_line
    assert validate_sql(result.output, "mysql").valid_syntax is True


def test_uuid_primary_key_does_not_become_an_integer_identity():
    result = compile_table_request("Create table accounts with id UUID primary key", "postgresql")

    assert result is not None
    assert '"id" UUID PRIMARY KEY' in result.output
    assert "GENERATED ALWAYS AS IDENTITY" not in result.output


def test_dbml_renders_relationships():
    result = compile_table_request(
        "Create table orders with customer_id integer references customers(id) in dbml",
        "dbml",
    )

    assert result is not None
    assert "Ref: orders.customer_id > customers.id" in result.output
    assert validate_sql(result.output, "dbml").valid_syntax is True


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


def test_multi_table_request_infers_conventional_foreign_keys():
    result = compile_ddl_request(
        "Create customers and orders tables with customer_id references customers(id)",
        "postgresql",
    )

    assert result is not None
    assert result.table_names == ["customers", "orders"]
    assert 'CREATE TABLE "customers"' in result.output
    assert 'CREATE TABLE "orders"' in result.output
    assert 'REFERENCES "customers" ("id")' in result.output
    assert result.plan["relationships"] == [
        {"from_table": "orders", "from_column": "customer_id", "to_table": "customers", "to_column": "id"}
    ]
    assert validate_sql(result.output, "postgresql").valid_syntax is True


def test_multi_table_request_orders_parent_before_child():
    result = compile_ddl_request("Create orders and customers tables", "postgresql")

    assert result is not None
    assert result.table_names == ["customers", "orders"]
    assert result.output.index('CREATE TABLE "customers"') < result.output.index('CREATE TABLE "orders"')
    assert 'REFERENCES "customers" ("customer_id")' in result.output


def test_multi_table_dbml_request_renders_relationships():
    result = compile_ddl_request("Create authors and books tables", "dbml")

    assert result is not None
    assert "Table authors {" in result.output
    assert "Table books {" in result.output
    assert "Ref: books.author_id > authors.author_id" in result.output
    assert validate_sql(result.output, "dbml").valid_syntax is True


def test_index_and_alter_requests_have_structured_plans():
    index = compile_ddl_request("Add a unique index named uq_orders_customer on orders(customer_id)", "mysql", "modify_ddl")
    alter = compile_ddl_request("Add a hire_date column to employees that defaults to current date", "postgresql", "modify_ddl")

    assert index is not None
    assert index.operation == "create_index"
    assert "CREATE UNIQUE INDEX `uq_orders_customer`" in index.output
    assert index.plan["columns"] == ["customer_id"]
    assert alter is not None
    assert alter.operation == "alter_add_column"
    assert "DEFAULT CURRENT_DATE" in alter.output
    assert alter.plan["columns"][0]["default"] == "CURRENT_DATE"


def test_alter_request_supports_multiple_natural_language_columns():
    result = compile_ddl_request(
        "Now add phone and date of birth columns to customer",
        "postgresql",
        "modify_ddl",
    )

    assert result is not None
    assert result.operation == "alter_add_columns"
    assert 'ADD COLUMN "phone" TEXT' in result.output
    assert 'ADD COLUMN "date_of_birth" DATE' in result.output
    assert [column["name"] for column in result.plan["columns"]] == ["phone", "date_of_birth"]


def test_natural_constraints_and_defaults_are_preserved():
    result = compile_table_request(
        "Create a table for products with a product name, non-negative price, and active boolean",
        "postgresql",
    )

    assert result is not None
    assert '"price" DECIMAL(12,2) CHECK ("price" >= 0)' in result.output
    assert '"active" BOOLEAN' in result.output
