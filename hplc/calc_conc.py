import math

def calculate_concentration(area, curve_params):
    """
    Given the calibration curve parameters, calculate the concentration based on peak area.
    """
    slope = curve_params['Slope']
    intercept = curve_params['Intercept']
    concentration = (area - intercept) / slope
    if not math.isnan(concentration):
        return round(concentration, 2)
    else:
        return None