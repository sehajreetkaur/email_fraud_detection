# reputation.py

import re
import dns.resolver
import whois
from datetime import datetime

# ── 1. EMAIL FORMAT CHECK ─────────────────────────────────────
def is_valid_email(email):
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w{2,}$'
    return bool(re.match(pattern, email))

# ── 2. FREE EMAIL PROVIDER CHECK ─────────────────────────────
FREE_PROVIDERS = {
    'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
    'icloud.com', 'aol.com', 'protonmail.com', 'mail.com',
    'gmx.com', 'yandex.com', 'live.com', 'msn.com'
}

TRUSTED_DOMAINS = {
    'apple.com', 'google.com', 'microsoft.com', 'amazon.com',
    'paypal.com', 'facebook.com', 'twitter.com', 'linkedin.com',
    'instagram.com', 'netflix.com', 'spotify.com', 'github.com',
    'dropbox.com', 'adobe.com', 'salesforce.com', 'zoom.us'
}

def check_free_provider(domain):
    return domain.lower() in FREE_PROVIDERS

def check_trusted_domain(domain):
    return domain.lower() in TRUSTED_DOMAINS

# ── 3. SPF RECORD CHECK ───────────────────────────────────────
def check_spf(domain):
    try:
        answers = dns.resolver.resolve(domain, 'TXT')
        for r in answers:
            if 'spf' in str(r).lower():
                return True
        return False
    except:
        return False

# ── 4. DMARC RECORD CHECK ─────────────────────────────────────
def check_dmarc(domain):
    try:
        answers = dns.resolver.resolve(f'_dmarc.{domain}', 'TXT')
        for r in answers:
            if 'dmarc' in str(r).lower():
                return True
        return False
    except:
        return False

# ── 5. MX RECORD CHECK ───────────────────────────────────────
def check_mx(domain):
    try:
        dns.resolver.resolve(domain, 'MX')
        return True
    except:
        return False

# ── 6. DOMAIN AGE CHECK ───────────────────────────────────────
def get_domain_age_days(domain):
    try:
        w = whois.whois(domain)
        creation = w.creation_date
        if isinstance(creation, list):
            creation = creation[0]
        if creation:
            age = (datetime.now() - creation).days
            return age
        return None
    except:
        return None

# ── 7. DISPLAY NAME MISMATCH CHECK ───────────────────────────
def check_display_name_mismatch(display_name, email_domain):
    if not display_name:
        return False
    display_lower = display_name.lower()
    domain_lower  = email_domain.lower().replace('.com','').replace('.org','')
    # Check if display name mentions a trusted brand but domain doesn't match
    for trusted in TRUSTED_DOMAINS:
        brand = trusted.replace('.com','').replace('.org','')
        if brand in display_lower and brand not in domain_lower:
            return True   # mismatch found!
    return False

# ── 8. MAIN REPUTATION SCORER ─────────────────────────────────
def analyze_sender(email_address, display_name=None):
    results = {
        'email':          email_address,
        'display_name':   display_name,
        'valid_format':   False,
        'domain':         None,
        'is_free':        False,
        'is_trusted':     False,
        'has_spf':        False,
        'has_dmarc':      False,
        'has_mx':         False,
        'domain_age':     None,
        'name_mismatch':  False,
        'risk_score':     0,
        'risk_level':     'Unknown',
        'flags':          []
    }

    # Format check
    if not is_valid_email(email_address):
        results['flags'].append('Invalid email format')
        results['risk_score'] = 100
        results['risk_level'] = 'High'
        return results

    results['valid_format'] = True
    domain = email_address.split('@')[1].lower()
    results['domain'] = domain

    # Run all checks
    results['is_free']    = check_free_provider(domain)
    results['is_trusted'] = check_trusted_domain(domain)
    results['has_spf']    = check_spf(domain)
    results['has_dmarc']  = check_dmarc(domain)
    results['has_mx']     = check_mx(domain)
    results['domain_age'] = get_domain_age_days(domain)
    results['name_mismatch'] = check_display_name_mismatch(
                                display_name or '', domain)

    # ── SCORING ───────────────────────────────────────────────
    score = 0

    if results['is_trusted']:
        score -= 30   # trusted domain = less risky
    if results['is_free']:
        score += 10   # free provider = slightly more suspicious

    if not results['has_spf']:
        score += 20
        results['flags'].append('No SPF record — domain has no mail policy')
    if not results['has_dmarc']:
        score += 20
        results['flags'].append('No DMARC record — emails unverified')
    if not results['has_mx']:
        score += 15
        results['flags'].append('No MX record — domain cannot receive emails')

    if results['domain_age'] is not None:
        if results['domain_age'] < 30:
            score += 40
            results['flags'].append(f'Domain is only {results["domain_age"]} days old — very new!')
        elif results['domain_age'] < 180:
            score += 20
            results['flags'].append(f'Domain is only {results["domain_age"]} days old — relatively new')
        elif results['domain_age'] < 365:
            score += 10

    if results['name_mismatch']:
        score += 35
        results['flags'].append(
            f'Display name "{display_name}" does not match domain "{domain}" — possible impersonation!')

    # Clamp score
    score = max(0, min(100, score))
    results['risk_score'] = score

    if score >= 60:
        results['risk_level'] = 'High'
    elif score >= 30:
        results['risk_level'] = 'Medium'
    else:
        results['risk_level'] = 'Low'

    return results