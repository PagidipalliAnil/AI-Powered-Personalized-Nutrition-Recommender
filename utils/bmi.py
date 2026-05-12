def calculate_bmi(height, weight):
    if height == 0 or weight == 0:
        return 0
    height_m = height / 100
    bmi = weight / (height_m ** 2)
    return round(bmi, 2)


def bmi_category(bmi):
    if bmi == 0:
        return "Invalid"
    if bmi < 18.5:
        return "Underweight"
    elif bmi < 25:
        return "Normal"
    elif bmi < 30:
        return "Overweight"
    else:
        return "Obese"


def ideal_weight_range(height):
    """Returns ideal weight range in kg for a given height in cm."""
    height_m = height / 100
    low = round(18.5 * (height_m ** 2), 1)
    high = round(24.9 * (height_m ** 2), 1)
    return low, high


def daily_calorie_needs(weight, height, age, gender, activity="moderate"):
    """Harris-Benedict equation for TDEE."""
    if gender.lower() == "male":
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:
        bmr = 10 * weight + 6.25 * height - 5 * age - 161

    multipliers = {
        "sedentary": 1.2,
        "light": 1.375,
        "moderate": 1.55,
        "active": 1.725,
        "very active": 1.9,
    }
    return round(bmr * multipliers.get(activity, 1.55))
