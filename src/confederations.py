# Strength multipliers used directly as a K-factor discount when the opponent
# is from this confederation. UEFA = 1.0 reference (top European leagues, strong
# at all levels). AFC/CAF heavily discounted because the strength range within
# them is huge — beating Kyrgyzstan or Liechtenstein-equivalent is less
# informative than beating an average European opponent. Tuned 2026-05-16 after
# observing Iran's Elo inflating above Belgium's via AFC qualifying farming.
CONFEDERATION_STRENGTH = {
    'UEFA':     1.00,
    'CONMEBOL': 0.90,
    'CONCACAF': 0.55,
    'AFC':      0.50,
    'CAF':      0.45,
    'OFC':      0.30,
}

TEAM_CONFEDERATION = {
    # UEFA
    'Albania': 'UEFA', 'Andorra': 'UEFA', 'Armenia': 'UEFA', 'Austria': 'UEFA',
    'Azerbaijan': 'UEFA', 'Belarus': 'UEFA', 'Belgium': 'UEFA', 'Bosnia and Herzegovina': 'UEFA',
    'Bulgaria': 'UEFA', 'Croatia': 'UEFA', 'Cyprus': 'UEFA', 'Czech Republic': 'UEFA',
    'Czechia': 'UEFA', 'Denmark': 'UEFA', 'England': 'UEFA', 'Estonia': 'UEFA',
    'Faroe Islands': 'UEFA', 'Finland': 'UEFA', 'France': 'UEFA', 'Georgia': 'UEFA',
    'Germany': 'UEFA', 'Gibraltar': 'UEFA', 'Greece': 'UEFA', 'Hungary': 'UEFA',
    'Iceland': 'UEFA', 'Ireland': 'UEFA', 'Republic of Ireland': 'UEFA', 'Israel': 'UEFA',
    'Italy': 'UEFA', 'Kazakhstan': 'UEFA', 'Kosovo': 'UEFA', 'Latvia': 'UEFA',
    'Liechtenstein': 'UEFA', 'Lithuania': 'UEFA', 'Luxembourg': 'UEFA', 'Malta': 'UEFA',
    'Moldova': 'UEFA', 'Montenegro': 'UEFA', 'Netherlands': 'UEFA', 'North Macedonia': 'UEFA',
    'Northern Ireland': 'UEFA', 'Norway': 'UEFA', 'Poland': 'UEFA', 'Portugal': 'UEFA',
    'Romania': 'UEFA', 'Russia': 'UEFA', 'San Marino': 'UEFA', 'Scotland': 'UEFA',
    'Serbia': 'UEFA', 'Slovakia': 'UEFA', 'Slovenia': 'UEFA', 'Spain': 'UEFA',
    'Sweden': 'UEFA', 'Switzerland': 'UEFA', 'Turkey': 'UEFA', 'Ukraine': 'UEFA',
    'Wales': 'UEFA',

    # CONMEBOL
    'Argentina': 'CONMEBOL', 'Bolivia': 'CONMEBOL', 'Brazil': 'CONMEBOL',
    'Chile': 'CONMEBOL', 'Colombia': 'CONMEBOL', 'Ecuador': 'CONMEBOL',
    'Paraguay': 'CONMEBOL', 'Peru': 'CONMEBOL', 'Uruguay': 'CONMEBOL',
    'Venezuela': 'CONMEBOL',

    # CONCACAF
    'Antigua and Barbuda': 'CONCACAF', 'Aruba': 'CONCACAF', 'Bahamas': 'CONCACAF',
    'Barbados': 'CONCACAF', 'Belize': 'CONCACAF', 'Bermuda': 'CONCACAF',
    'Canada': 'CONCACAF', 'Cayman Islands': 'CONCACAF', 'Costa Rica': 'CONCACAF',
    'Cuba': 'CONCACAF', 'Curaçao': 'CONCACAF', 'Dominican Republic': 'CONCACAF',
    'El Salvador': 'CONCACAF', 'Grenada': 'CONCACAF', 'Guatemala': 'CONCACAF',
    'Guyana': 'CONCACAF', 'Haiti': 'CONCACAF', 'Honduras': 'CONCACAF',
    'Jamaica': 'CONCACAF', 'Mexico': 'CONCACAF', 'Nicaragua': 'CONCACAF',
    'Panama': 'CONCACAF', 'Puerto Rico': 'CONCACAF', 'Saint Kitts and Nevis': 'CONCACAF',
    'Saint Lucia': 'CONCACAF', 'Saint Vincent and the Grenadines': 'CONCACAF',
    'Suriname': 'CONCACAF', 'Trinidad and Tobago': 'CONCACAF',
    'United States': 'CONCACAF', 'USA': 'CONCACAF',

    # AFC
    'Afghanistan': 'AFC', 'Australia': 'AFC', 'Bahrain': 'AFC', 'Bangladesh': 'AFC',
    'Bhutan': 'AFC', 'Brunei': 'AFC', 'Cambodia': 'AFC', 'China': 'AFC',
    'China PR': 'AFC', 'Chinese Taipei': 'AFC', 'Guam': 'AFC', 'Hong Kong': 'AFC',
    'India': 'AFC', 'Indonesia': 'AFC', 'Iran': 'AFC', 'Iraq': 'AFC',
    'Japan': 'AFC', 'Jordan': 'AFC', 'Kuwait': 'AFC', 'Kyrgyzstan': 'AFC',
    'Laos': 'AFC', 'Lebanon': 'AFC', 'Macau': 'AFC', 'Malaysia': 'AFC',
    'Maldives': 'AFC', 'Mongolia': 'AFC', 'Myanmar': 'AFC', 'Nepal': 'AFC',
    'North Korea': 'AFC', 'Oman': 'AFC', 'Pakistan': 'AFC', 'Palestine': 'AFC',
    'Philippines': 'AFC', 'Qatar': 'AFC', 'Saudi Arabia': 'AFC', 'Singapore': 'AFC',
    'South Korea': 'AFC', 'Sri Lanka': 'AFC', 'Syria': 'AFC', 'Tajikistan': 'AFC',
    'Thailand': 'AFC', 'Timor-Leste': 'AFC', 'Turkmenistan': 'AFC',
    'United Arab Emirates': 'AFC', 'Uzbekistan': 'AFC', 'Vietnam': 'AFC',
    'Yemen': 'AFC',

    # CAF
    'Algeria': 'CAF', 'Angola': 'CAF', 'Benin': 'CAF', 'Botswana': 'CAF',
    'Burkina Faso': 'CAF', 'Burundi': 'CAF', 'Cameroon': 'CAF',
    'Cape Verde': 'CAF', 'Central African Republic': 'CAF', 'Chad': 'CAF',
    'Comoros': 'CAF', 'DR Congo': 'CAF', 'Congo': 'CAF', 'Djibouti': 'CAF',
    'Egypt': 'CAF', 'Equatorial Guinea': 'CAF', 'Eritrea': 'CAF',
    'Eswatini': 'CAF', 'Ethiopia': 'CAF', 'Gabon': 'CAF', 'Gambia': 'CAF',
    'Ghana': 'CAF', 'Guinea': 'CAF', 'Guinea-Bissau': 'CAF', 'Ivory Coast': 'CAF',
    'Kenya': 'CAF', 'Lesotho': 'CAF', 'Liberia': 'CAF', 'Libya': 'CAF',
    'Madagascar': 'CAF', 'Malawi': 'CAF', 'Mali': 'CAF', 'Mauritania': 'CAF',
    'Mauritius': 'CAF', 'Morocco': 'CAF', 'Mozambique': 'CAF', 'Namibia': 'CAF',
    'Niger': 'CAF', 'Nigeria': 'CAF', 'Rwanda': 'CAF', 'São Tomé and Príncipe': 'CAF',
    'Senegal': 'CAF', 'Sierra Leone': 'CAF', 'Somalia': 'CAF', 'South Africa': 'CAF',
    'South Sudan': 'CAF', 'Sudan': 'CAF', 'Tanzania': 'CAF', 'Togo': 'CAF',
    'Tunisia': 'CAF', 'Uganda': 'CAF', 'Zambia': 'CAF', 'Zimbabwe': 'CAF',

    # OFC
    'American Samoa': 'OFC', 'Cook Islands': 'OFC', 'Fiji': 'OFC',
    'New Caledonia': 'OFC', 'New Zealand': 'OFC', 'Papua New Guinea': 'OFC',
    'Samoa': 'OFC', 'Solomon Islands': 'OFC', 'Tahiti': 'OFC', 'Tonga': 'OFC',
    'Vanuatu': 'OFC',
}

def get_confederation(team: str) -> str:
    return TEAM_CONFEDERATION.get(team, 'UEFA')  # default to UEFA if unknown

def confederation_k_multiplier(opponent: str) -> float:
    """Reduce K based on the opponent's confederation strength.

    Returned values: UEFA 1.00, CONMEBOL 0.90, CONCACAF 0.55, AFC 0.50,
    CAF 0.45, OFC 0.30. Lower multiplier = less Elo update per match.
    Rationale: beating a weak-confederation opponent is less informative
    about your true skill than beating a UEFA opponent of equivalent rank.
    """
    return CONFEDERATION_STRENGTH[get_confederation(opponent)]
