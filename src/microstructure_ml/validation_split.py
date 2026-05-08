from pathlib import Path

def list_dates(input_base: Path) -> list[Path]:
    date_dirs = sorted(
        (p for p in input_base.rglob("date=*") if p.is_dir()), 
        key=lambda p: p.name
        )
    return date_dirs

def train_val_splits(dates: list, val_days: int, step_days: int) -> list:
    splits = []
    for i in range(1, len(dates) - val_days + 1, step_days):
        splits.append((dates[:i], dates[i:i+val_days]))
    return splits

def create_splits(input_base: Path, val_days: int, step_days: int, test_days: int) -> tuple[list, list]:
    dates = list_dates(input_base)
    if test_days + val_days + 1 > len(dates):
        raise ValueError("Insufficient data for specified training or testing amount")
    test_dates = dates[len(dates) - test_days:]
    train_val_splits_input_dates = dates[:len(dates) - test_days]
    train_val_dates = train_val_splits(train_val_splits_input_dates, val_days, step_days)
    return train_val_dates, test_dates
    

    


