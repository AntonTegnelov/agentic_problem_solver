"""Tests for the main entry point."""


def test_main_entry_point() -> None:
    """Test that the main entry point exists."""
    # Just verify that the module can be imported
    import src.__main__

    # Check that it has the expected attributes
    assert hasattr(src.__main__, "main")

    # We don't actually call main() since that would execute the CLI
