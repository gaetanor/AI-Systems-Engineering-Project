"""
src/core/guardrails.py

Security & Privacy Module.
Implements PII (Personally Identifiable Information) detection and masking logic.
Acts as a firewall before the query reaches the LLM.
Updated to handle SSN and malformed inputs (spaces, dashes, dots).
"""

import re
from typing import Tuple

class PIIGuardrail:
    """
    Detects and handles sensitive information in user queries.
    Supported Patterns: Email, Phone, Credit Cards, IBAN, SSN (US).
    """

    # Regex Patterns Compilati
    PATTERNS = {
        # EMAIL: Standard email format
        "EMAIL": re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        ),
        
        # PHONE: Handles (123) 456-7890, 123-456-7890, 123.456.7890, +1 123 456 7890
        "PHONE": re.compile(
            r'\b(?:\+?\d{1,3}[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b'
        ),
        
        # CREDIT_CARD: 13 to 19 digits, allowing spaces, dashes or dots as separators.
        # Es: 4532 1234 5678 9012 or 4532-1234-5678-9012 or 4532123456789012
        "CREDIT_CARD": re.compile(
            r'\b(?:\d[ -.]{0,1}){13,19}\b'
        ),
        
        # SSN (US): AAA-GG-SSSS or AAA GG SSSS or AAA.GG.SSSS
        "SSN": re.compile(
            r'\b\d{3}[-. ]?\d{2}[-. ]?\d{4}\b'
        ),
        
        # IBAN: International Bank Account Number
        "IBAN": re.compile(
            r'\b[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}\b'
        )
    }

    @staticmethod
    def scan_and_redact(text: str) -> Tuple[bool, str, list]:
        """
        Scans text for PII.
        Returns:
        - is_safe (bool): False if PII found
        - sanitized_text (str): Text with PII replaced by [REDACTED]
        - detected_types (list): List of PII types found (e.g., ['EMAIL'])
        """
        sanitized_text = text
        detected_types = []
        found_pii = False

        # Iteriamo su tutti i pattern
        for pii_type, pattern in PIIGuardrail.PATTERNS.items():
            # Troviamo tutte le occorrenze
            matches = pattern.findall(sanitized_text)
            if matches:
                # Filtraggio Extra per Carte di Credito:
                # Evitiamo di censurare anni o timestamp semplici (se < 13 cifre reali)
                if pii_type == "CREDIT_CARD":
                    valid_matches = []
                    for m in matches:
                        # Contiamo solo le cifre effettive
                        digit_count = sum(c.isdigit() for c in m)
                        if digit_count >= 13:
                            valid_matches.append(m)
                    
                    if not valid_matches:
                        continue # Se erano falsi positivi, saltiamo
                
                found_pii = True
                if pii_type not in detected_types:
                    detected_types.append(pii_type)
                
                # Sostituzione
                # Usiamo una funzione lambda per sostituire ogni match trovato
                sanitized_text = pattern.sub(f"[{pii_type}_REDACTED]", sanitized_text)

        return not found_pii, sanitized_text, detected_types