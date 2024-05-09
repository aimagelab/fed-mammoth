def str_to_bool(s: str) -> bool:
    return s.lower() in ["true", "1", "t", "y", "yes"]


def get_time_str(delta_time: int):
    delta_seconds = int(delta_time % 60)
    delta_minutes = int((delta_time // 60) % 60)
    delta_hours = int((delta_time // 3600) % 24)
    delta_days = int(delta_time // (24 * 3600))

    delta_time_str = ""
    for remaining, unity in zip([delta_days, delta_hours, delta_minutes], ["d", "h", "m"]):
        if remaining > 0:
            delta_time_str += f" {remaining}{unity}"
    if delta_days == 0 and delta_hours == 0:
        delta_time_str += f" {delta_seconds}s"
    return delta_time_str[1:]
