import os

from agentq.config.config import USER_PREFERENCES_PATH
from agentq.utils.logger import logger


def get_user_ltm():
    user_preference_file_name = "user_preferences.txt"
    user_preference_file = os.path.join(
        USER_PREFERENCES_PATH, user_preference_file_name
    )
    try:
        with open(user_preference_file) as file:
            user_pref = file.read()
        return user_pref
    except FileNotFoundError:
        logger.warning(f"User preference file not found: {user_preference_file}")

    return None
