from typing import Dict


config: Dict[str, list] = {}

config['engagement_features'] = [
    'clumpiness_score', 'norm_dwell_time', 'norm_pi_per_visit',
    'norm_offer_page_seen', 'norm_paywall_seen',
    'norm_registerlogin_page_seen', 'norm_pv_of_article',
    'recency', 'frequency', 'Friday', 'Monday', 'Saturday',
    'Sunday', 'Thursday', 'Tuesday', 'Wednesday', 'Evening',
    'Morning', 'Night', 'Noon', 'bounce_rate', 'user_login',
    'auto', 'bundesliga', 'erotik', 'fussball', 'geld', 'horoskop',
    'lifestyle', 'other_genre', 'politik',
    'ratgeber', 'regional', 'spiele', 'sport', 'unterhaltung'
]

config['non_engagement_features'] = [
    'price_1','price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7',
    'amazon', 'apple', 'google', 'microsoft','mozilla', 'opera',
    'other', 'other_browser', 'samsung', 'desktop', 'austria',
    'baden-wurttemberg ', 'bayern ', 'berlin ', 'thuringen ',
    'brandenburg ', 'bremen ', 'hamburg ', 'hessen ', 'luxembourg',
    'mecklenburg-vorpommern ','niedersachsen ', 'nordrhein-westfalen ',
    'other_country', 'rheinland-pfalz ', 'saarland ', 'sachsen ',
    'sachsen-anhalt ', 'schleswig-holstein ', 'switzerland',
]

config['all_features'] = [
    'user_login', 'bounce_rate', 'clumpiness_score',
    'norm_dwell_time', 'norm_offer_page_seen', 'norm_paywall_seen',
    'norm_registerlogin_page_seen', 'norm_pi_per_visit',
    'norm_pv_of_article', 'recency', 'frequency', 'desktop', 'price_1',
    'price_2', 'price_3', 'price_4', 'price_5', 'price_6', 'price_7',
    'Friday', 'Monday','Saturday', 'Sunday', 'Thursday', 'Tuesday',
    'Wednesday', 'Evening', 'Morning', 'Night', 'Noon', 'amazon',
    'apple', 'google', 'microsoft', 'mozilla', 'opera', 'other_browser',
    'samsung', 'auto', 'bundesliga', 'erotik', 'fussball', 'geld', 'horoskop',
    'lifestyle','other_genre', 'politik', 'ratgeber', 'regional', 'spiele', 'sport',
    'unterhaltung', 'austria','baden-wurttemberg ', 'bayern ', 'berlin ',
    'brandenburg ', 'bremen ', 'hamburg ', 'hessen ', 'luxembourg',
    'mecklenburg-vorpommern ','niedersachsen ', 'nordrhein-westfalen ',
    'other_country', 'rheinland-pfalz ', 'saarland ', 'sachsen ',
    'sachsen-anhalt ', 'schleswig-holstein ', 'switzerland', 'thuringen '
]

config['genre'] = [
    'regional', 'politik', 'unterhaltung', 'sport', 'ratgeber',
    'erotik', 'lifestyle', 'geld', 'other_genre', 'bundesliga',
    'fussball', 'spiele', 'auto', 'horoskop'
]

config['day_of_week'] = [
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday',
    'Saturday', 'Sunday'
]

config['day_time'] = [
    'Night', 'Morning', 'Evening', 'Noon'
]

config['mc_columns'] = [
    'norm_dwell_time', 'norm_offer_page_seen',
    'norm_paywall_seen', 'norm_registerlogin_page_seen',
    'norm_pv_of_article', 'recency', 'frequency',
    'norm_pi_per_visit'
]
