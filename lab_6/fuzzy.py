import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

# Define input variables
moisture = ctrl.Antecedent(np.arange(0, 11, 1), 'moisture')
sensitivity = ctrl.Antecedent(np.arange(0, 11, 1), 'sensitivity')
storage_time = ctrl.Antecedent(np.arange(0, 11, 1), 'storage_time')

# Define output variable
shelf = ctrl.Consequent(np.arange(0, 101, 1), 'shelf')  # 0-100 range

# Membership functions for inputs
moisture['low'] = fuzz.trimf(moisture.universe, [0, 0, 4])
moisture['medium'] = fuzz.trimf(moisture.universe, [2, 5, 8])
moisture['high'] = fuzz.trimf(moisture.universe, [6, 10, 10])

sensitivity['low'] = fuzz.trimf(sensitivity.universe, [0, 0, 4])
sensitivity['medium'] = fuzz.trimf(sensitivity.universe, [2, 5, 8])
sensitivity['high'] = fuzz.trimf(sensitivity.universe, [6, 10, 10])

storage_time['short'] = fuzz.trimf(storage_time.universe, [0, 0, 4])
storage_time['medium'] = fuzz.trimf(storage_time.universe, [2, 5, 8])
storage_time['long'] = fuzz.trimf(storage_time.universe, [6, 10, 10])

# Membership functions for output (shelf types in increasing value)
shelf['top_shelf'] = fuzz.trimf(shelf.universe, [0, 5, 10])
shelf['middle_shelf'] = fuzz.trimf(shelf.universe, [10, 20, 30])
shelf['lower_shelf'] = fuzz.trimf(shelf.universe, [30, 40, 50])
shelf['crisper_left'] = fuzz.trimf(shelf.universe, [50, 55, 60])
shelf['crisper_right'] = fuzz.trimf(shelf.universe, [60, 65, 70])
shelf['door_shelves'] = fuzz.trimf(shelf.universe, [70, 75, 80])
shelf['freezer_upper'] = fuzz.trimf(shelf.universe, [80, 83, 86])
shelf['freezer_middle'] = fuzz.trimf(shelf.universe, [86, 90, 94])
shelf['freezer_bottom'] = fuzz.trimf(shelf.universe, [94, 97, 100])

# Define example rules (expand as needed)
rules = [
    ctrl.Rule(moisture['low'] & sensitivity['low'] & storage_time['short'], shelf['top_shelf']),
    ctrl.Rule(moisture['low'] & sensitivity['low'] & storage_time['medium'], shelf['middle_shelf']),
    ctrl.Rule(moisture['low'] & sensitivity['low'] & storage_time['long'], shelf['freezer_upper']),

    ctrl.Rule(moisture['low'] & sensitivity['medium'] & storage_time['short'], shelf['middle_shelf']),
    ctrl.Rule(moisture['low'] & sensitivity['medium'] & storage_time['medium'], shelf['lower_shelf']),
    ctrl.Rule(moisture['low'] & sensitivity['medium'] & storage_time['long'], shelf['freezer_middle']),

    ctrl.Rule(moisture['low'] & sensitivity['high'] & storage_time['short'], shelf['door_shelves']),
    ctrl.Rule(moisture['low'] & sensitivity['high'] & storage_time['medium'], shelf['lower_shelf']),
    ctrl.Rule(moisture['low'] & sensitivity['high'] & storage_time['long'], shelf['freezer_bottom']),

    ctrl.Rule(moisture['medium'] & sensitivity['low'] & storage_time['short'], shelf['middle_shelf']),
    ctrl.Rule(moisture['medium'] & sensitivity['low'] & storage_time['medium'], shelf['lower_shelf']),
    ctrl.Rule(moisture['medium'] & sensitivity['low'] & storage_time['long'], shelf['freezer_upper']),

    ctrl.Rule(moisture['medium'] & sensitivity['medium'] & storage_time['short'], shelf['lower_shelf']),
    ctrl.Rule(moisture['medium'] & sensitivity['medium'] & storage_time['medium'], shelf['crisper_left']),
    ctrl.Rule(moisture['medium'] & sensitivity['medium'] & storage_time['long'], shelf['freezer_middle']),

    ctrl.Rule(moisture['medium'] & sensitivity['high'] & storage_time['short'], shelf['crisper_right']),
    ctrl.Rule(moisture['medium'] & sensitivity['high'] & storage_time['medium'], shelf['crisper_right']),
    ctrl.Rule(moisture['medium'] & sensitivity['high'] & storage_time['long'], shelf['freezer_bottom']),

    ctrl.Rule(moisture['high'] & sensitivity['low'] & storage_time['short'], shelf['lower_shelf']),
    ctrl.Rule(moisture['high'] & sensitivity['low'] & storage_time['medium'], shelf['crisper_left']),
    ctrl.Rule(moisture['high'] & sensitivity['low'] & storage_time['long'], shelf['freezer_upper']),

    ctrl.Rule(moisture['high'] & sensitivity['medium'] & storage_time['short'], shelf['crisper_left']),
    ctrl.Rule(moisture['high'] & sensitivity['medium'] & storage_time['medium'], shelf['crisper_right']),
    ctrl.Rule(moisture['high'] & sensitivity['medium'] & storage_time['long'], shelf['freezer_middle']),

    ctrl.Rule(moisture['high'] & sensitivity['high'] & storage_time['short'], shelf['crisper_right']),
    ctrl.Rule(moisture['high'] & sensitivity['high'] & storage_time['medium'], shelf['freezer_middle']),
    ctrl.Rule(moisture['high'] & sensitivity['high'] & storage_time['long'], shelf['freezer_bottom']),
]

# Create control system and simulation
shelf_ctrl = ctrl.ControlSystem(rules)
shelf_sim = ctrl.ControlSystemSimulation(shelf_ctrl)

def interpret_shelf(value):
    shelf_labels = [
        (0, 10, "Top Shelf"),
        (10, 30, "Middle Shelf"),
        (30, 50, "Lower Shelf"),
        (50, 60, "Crisper Drawer (Left)"),
        (60, 70, "Crisper Drawer (Right)"),
        (70, 80, "Door Shelves"),
        (80, 86, "Upper Freezer Drawers"),
        (86, 94, "Middle Freezer Drawers"),
        (94, 101, "Bottom Freezer Drawers")
    ]
    return next(label for low, high, label in shelf_labels if low <= value < high)