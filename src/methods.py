from datetime import datetime


def get_timestamp():
    return (datetime.now().__str__()
            .replace(" ", "_")
            .replace(":", "")
            .replace("-", "")
            .replace(".", "-")[:-3])


