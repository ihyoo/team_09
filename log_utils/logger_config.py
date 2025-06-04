"""
log config
"""
import logging
from logging.handlers import TimedRotatingFileHandler
import os
from datetime import datetime

class MonthlyTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, log_path, *args, **kwargs):
        # log_path: /.../log_utils/logs
        self.base_log_path = log_path
        os.makedirs(self.base_log_path, exist_ok=True)

        # 현재 날짜 기반 월 폴더와 파일 이름 생성
        today = datetime.now()
        self.month_dir = today.strftime("%Y-%m")
        self.date_str = today.strftime("%Y-%m-%d")
        full_dir = os.path.join(self.base_log_path, self.month_dir)
        os.makedirs(full_dir, exist_ok=True)

        file_path = os.path.join(full_dir, f"app_{self.date_str}.log")

        super().__init__(file_path, when="midnight", interval=1, backupCount=30, encoding="utf-8")

    def doRollover(self):
        # 다음날로 넘어갈 때 새로운 경로로 변경
        today = datetime.now()
        self.month_dir = today.strftime("%Y-%m")
        self.date_str = today.strftime("%Y-%m-%d")
        full_dir = os.path.join(self.base_log_path, self.month_dir)
        os.makedirs(full_dir, exist_ok=True)

        self.baseFilename = os.path.join(full_dir, f"app_{self.date_str}.log")
        super().doRollover()


def get_logger(name: str = "default_logger") -> logging.Logger:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_root_dir = os.path.join(base_dir, "logs")

    logger = logging.getLogger(name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        handler = MonthlyTimedRotatingFileHandler(log_root_dir)

        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.propagate = False

    return logger

