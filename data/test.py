import re

def extract_snomed_codes(text):
    """
    Extract all SNOMED CT codes from a given piece of text.
    SNOMED CT codes are typically numeric strings (6-18 digits).
    Returns a list of unique SNOMED codes found in the text.
    """
    # First, try to extract from <answer> tags if present
    answer_match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    search_text = answer_match.group(1).strip() if answer_match else text
    
    # Find all numeric sequences that could be SNOMED codes
    # SNOMED CT codes are typically 6-18 digits long
    # Pattern: sequences of digits, potentially with commas/spaces around them
    snomed_pattern = r'\b\d{6,18}\b'
    matches = re.findall(snomed_pattern, search_text)
    
    # Return unique codes (as strings)
    return list(set(matches))


if __name__ == "__main__":
    # Test case
    test_text = "<answer> Metipranolol hydrochloride (126152000, 126153000, 126154005) </answer>"
    
    print("Test text:")
    print(test_text)
    print("\nExtracted SNOMED codes:")
    codes = extract_snomed_codes(test_text)
    print(codes)
    print(f"\nNumber of codes found: {len(codes)}")
    print(f"Expected codes: ['126152000', '126153000', '126154005']")
    
    # Verify expected codes
    expected = {'126152000', '126153000', '126154005'}
    found = set(codes)
    if found == expected:
        print("\n✓ Test PASSED: All expected codes found!")
    else:
        print(f"\n✗ Test FAILED:")
        print(f"  Missing: {expected - found}")
        print(f"  Extra: {found - expected}")