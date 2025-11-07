import logging

from nonconform.utils.func.logger import get_logger


class TestLoggerCreation:
    def test_creates_logger_with_correct_name(self):
        logger = get_logger("test_module")
        assert logger.name == "nonconform.test_module"

    def test_returns_logger_instance(self):
        logger = get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_different_names_create_different_loggers(self):
        logger1 = get_logger("module1")
        logger2 = get_logger("module2")
        assert logger1.name != logger2.name

    def test_same_name_returns_same_logger(self):
        logger1 = get_logger("same_module")
        logger2 = get_logger("same_module")
        assert logger1 is logger2


class TestLoggerConfiguration:
    def test_logger_has_handler(self):
        get_logger("test_handler")
        root_logger = logging.getLogger("nonconform")
        assert len(root_logger.handlers) > 0

    def test_root_logger_propagate_is_false(self):
        get_logger("test_propagate")
        root_logger = logging.getLogger("nonconform")
        assert root_logger.propagate is False

    def test_default_level_is_info(self):
        get_logger("test_level")
        root_logger = logging.getLogger("nonconform")
        assert root_logger.level == logging.INFO

    def test_no_duplicate_handlers_on_multiple_calls(self):
        get_logger("test_duplicate")
        handler_count1 = len(logging.getLogger("nonconform").handlers)

        get_logger("test_duplicate_2")
        handler_count2 = len(logging.getLogger("nonconform").handlers)

        assert handler_count1 == handler_count2


class TestLoggingBehavior:
    def test_can_log_info_message(self, capture_logs):
        logger = get_logger("test_info")
        capture_logs.attach(logger)

        logger.info("Test info message")
        messages = capture_logs.get_messages(logging.INFO)
        assert "Test info message" in messages

        capture_logs.detach(logger)

    def test_can_log_warning_message(self, capture_logs):
        logger = get_logger("test_warning")
        capture_logs.attach(logger)

        logger.warning("Test warning message")
        messages = capture_logs.get_messages(logging.WARNING)
        assert "Test warning message" in messages

        capture_logs.detach(logger)

    def test_can_log_error_message(self, capture_logs):
        logger = get_logger("test_error")
        capture_logs.attach(logger)

        logger.error("Test error message")
        messages = capture_logs.get_messages(logging.ERROR)
        assert "Test error message" in messages

        capture_logs.detach(logger)

    def test_debug_not_logged_by_default(self, capture_logs):
        logger = get_logger("test_debug")
        capture_logs.attach(logger)

        logger.debug("Test debug message")
        messages = capture_logs.get_messages(logging.DEBUG)
        assert len(messages) == 0

        capture_logs.detach(logger)


class TestMultipleLoggers:
    def test_multiple_loggers_dont_interfere(self, capture_logs):
        logger1 = get_logger("module_a")
        logger2 = get_logger("module_b")

        capture_logs.attach(logger1)
        logger1.info("Message from module A")
        messages = capture_logs.get_messages()
        assert "Message from module A" in messages
        capture_logs.detach(logger1)

        capture_logs.clear()
        capture_logs.attach(logger2)
        logger2.info("Message from module B")
        messages = capture_logs.get_messages()
        assert "Message from module B" in messages
        capture_logs.detach(logger2)

    def test_loggers_can_have_different_levels(self):
        logger1 = get_logger("level_test_1")
        logger2 = get_logger("level_test_2")

        logger1.setLevel(logging.DEBUG)
        logger2.setLevel(logging.WARNING)

        assert logger1.level == logging.DEBUG
        assert logger2.level == logging.WARNING


class TestEdgeCases:
    def test_empty_name_string(self):
        logger = get_logger("")
        assert logger.name == "nonconform."

    def test_name_with_dots(self):
        logger = get_logger("module.submodule")
        assert logger.name == "nonconform.module.submodule"

    def test_name_with_underscores(self):
        logger = get_logger("test_module_name")
        assert logger.name == "nonconform.test_module_name"

    def test_very_long_name(self):
        long_name = "a" * 200
        logger = get_logger(long_name)
        assert logger.name == f"nonconform.{long_name}"
