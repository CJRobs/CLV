# config.py
"""
Configuration settings for CLV analysis
"""

# Model parameters
CLV_CONFIG = {
    'profit_margin': 0.2,  # 20% profit margin
    'discount_rate': 0.05,  # 5% annual discount rate
    'time_period': 12,     # months
    'sample_size': None,   # None for all customers, or specify number
}

# Segment lifespan multipliers
LIFESPAN_MULTIPLIERS = {
    'Champions': 5,
    'Loyal Customers': 4,
    'Potential Loyalists': 3,
    'New Customers': 3,
    'Promising': 2.5,
    'Need Attention': 2,
    'About to Pause': 1.5,
    'At Risk': 1
}

# BG/NBD Model convergence settings
BGNBD_PENALIZERS = [0.0, 0.001, 0.01, 0.1, 1.0]

# Segment definitions for RFM scoring
SEGMENT_RULES = {
    'Champions': ['55', '54', '45'],
    'Loyal Customers': ['53', '44', '35', '43'],
    'Potential Loyalists': ['52', '42', '33', '34'],
    'New Customers': ['51', '41', '31', '32'],
    'Promising': ['25', '24', '15'],
    'Need Attention': ['23', '22', '13', '14'],
    'About to Pause': ['21', '12', '11']
}

# Mapping for categorical variables
MAPPINGS = {
    'age': {
        '0-17': 1, '18-25': 2, '26-35': 3, 
        '36-45': 4, '46-50': 5, '51-55': 6, '55+': 7
    },
    'city': {'A': 3, 'B': 2, 'C': 1},
    'stay_years': {'0': 0, '1': 1, '2': 2, '3': 3, '4+': 4}
}

# Visualization settings
VIZ_CONFIG = {
    'figure_size': (15, 12),
    'color_palette': 'viridis',
    'style': 'seaborn-darkgrid'
}

# Recommendations by segment
SEGMENT_RECOMMENDATIONS = {
    'Champions': 'VIP treatment, early access to new products, exclusive rewards',
    'Loyal Customers': 'Loyalty programs, referral incentives, personalized offers',
    'Potential Loyalists': 'Targeted promotions, engagement campaigns, product recommendations',
    'New Customers': 'Welcome series, onboarding support, first-purchase incentives',
    'Promising': 'Limited-time offers, category expansion incentives',
    'Need Attention': 'Re-engagement campaigns, satisfaction surveys, win-back offers',
    'About to Sleep': 'Urgent reactivation offers, feedback requests',
    'At Risk': 'High-value incentives, personalized outreach, churn prevention'
}