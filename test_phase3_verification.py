"""
Acceptance Test Suite for Phase 3 — Claim Verification
"""
import sys
import claim_verifier

def run_tests():
    print("=== RUNNING PHASE 3 ACCEPTANCE TESTS ===")
    
    # Test 1: Deterministic math (correct)
    c1 = claim_verifier.extract_claims("2 + 2 = 4")[0]
    r1 = claim_verifier.verify_single_claim(c1, "run_001")
    assert r1["verification_status"] == claim_verifier.VerificationStatus.SUPPORTED, f"Expected SUPPORTED, got {r1['verification_status']}"
    assert r1["verification_method"] == claim_verifier.VerificationMethod.DETERMINISTIC_CALCULATION
    print("PASS: 1. Deterministic math (2+2=4) -> SUPPORTED")

    # Test 2: Numerical comparison (contradicted)
    c2 = claim_verifier.extract_claims("9.11 is larger than 9.9")[0]
    r2 = claim_verifier.verify_single_claim(c2, "run_001")
    assert r2["verification_status"] == claim_verifier.VerificationStatus.CONTRADICTED, f"Expected CONTRADICTED, got {r2['verification_status']}"
    assert "Calculated" in r2["relevant_excerpt"]
    print("PASS: 2. Decimal comparison (9.11 is larger than 9.9) -> CONTRADICTED")

    # Test 3: Numerical comparison (supported)
    c3 = claim_verifier.extract_claims("9.9 is larger than 9.11")[0]
    r3 = claim_verifier.verify_single_claim(c3, "run_001")
    assert r3["verification_status"] == claim_verifier.VerificationStatus.SUPPORTED, f"Expected SUPPORTED, got {r3['verification_status']}"
    print("PASS: 3. Decimal comparison (9.9 is larger than 9.11) -> SUPPORTED")

    # Test 4: Demonstrably incorrect definition / factual statement
    c4 = claim_verifier.extract_claims("Rome is the capital of France.")[0]
    r4 = claim_verifier.verify_single_claim(c4, "run_001")
    assert r4["verification_status"] == claim_verifier.VerificationStatus.CONTRADICTED, f"Expected CONTRADICTED, got {r4['verification_status']}"
    assert r4["source_url"] != ""
    print(f"PASS: 4. Incorrect factual statement (Rome is capital of France) -> CONTRADICTED with source: {r4['source_title']}")

    # Test 5: Supported definition
    c5 = claim_verifier.extract_claims("Artificial intelligence is a branch of computer science.")[0]
    r5 = claim_verifier.verify_single_claim(c5, "run_001")
    assert r5["verification_status"] == claim_verifier.VerificationStatus.SUPPORTED, f"Expected SUPPORTED, got {r5['verification_status']}"
    print(f"PASS: 5. Supported definition (AI is branch of CS) -> SUPPORTED with source: {r5['source_title']}")

    # Test 6: Subjective opinion statement
    c6 = claim_verifier.extract_claims("Python is the best programming language in the world.")[0]
    r6 = claim_verifier.verify_single_claim(c6, "run_001")
    assert r6["verification_status"] == claim_verifier.VerificationStatus.NOT_VERIFIABLE, f"Expected NOT_VERIFIABLE, got {r6['verification_status']}"
    assert r6["claim_type"] == claim_verifier.ClaimType.OPINION
    print("PASS: 6. Subjective opinion -> NOT_VERIFIABLE (Properly filtered)")

    # Test 7: Obscure claim with insufficient evidence (never fabricate sources!)
    c7 = claim_verifier.extract_claims("The ancient king of Galdoria had 42 zebras in 1204 BC.")[0]
    r7 = claim_verifier.verify_single_claim(c7, "run_001")
    assert r7["verification_status"] == claim_verifier.VerificationStatus.INSUFFICIENT_EVIDENCE, f"Expected INSUFFICIENT_EVIDENCE, got {r7['verification_status']}"
    assert r7["source_url"] == "", "Citation must not be fabricated when evidence is unavailable"
    print("PASS: 7. Obscure claim -> INSUFFICIENT_EVIDENCE (No hallucinated sources)")

    # Test 8: Truncated response handling
    truncated_text = "Artificial intelligence was founded in 1956 and can process large amounts of data such as"
    claims_trunc = claim_verifier.extract_claims(truncated_text, is_truncated=True)
    last_claim = claims_trunc[-1]
    r8 = claim_verifier.verify_single_claim(last_claim, "run_001")
    assert r8["is_from_truncated_output"] is True
    assert r8["verification_status"] == claim_verifier.VerificationStatus.NOT_VERIFIABLE
    assert "truncated" in r8["explanation"].lower()
    print("PASS: 8. Truncated output claim -> Flagged & Not Verifiable without silent reconstruction")

    # Test 9: Complete Pipeline execution
    multi_text = (
        "Artificial intelligence is a branch of computer science. "
        "2 + 2 = 4. "
        "Python is the best language. "
        "9.11 is larger than 9.9."
    )
    res_batch = claim_verifier.execute_phase3_verification(multi_text, "run_test_batch", prompt="Explain AI and math", is_truncated=False)
    assert res_batch["total_claims"] == 4
    assert res_batch["status_summary"][claim_verifier.VerificationStatus.SUPPORTED] >= 2
    assert res_batch["status_summary"][claim_verifier.VerificationStatus.CONTRADICTED] >= 1
    assert res_batch["status_summary"][claim_verifier.VerificationStatus.NOT_VERIFIABLE] >= 1
    print(f"PASS: 9. Multi-claim pipeline executed successfully with {res_batch['total_claims']} claims:")
    for c in res_batch["claims"]:
        print(f"   [{c['verification_status']}] ({c['claim_type']}) {c['exact_claim_text']}")

    # Test 10: Original answer untouched & offsets valid
    for c in res_batch["claims"]:
        start = c["location"]["start_char"]
        end = c["location"]["end_char"]
        extracted_slice = multi_text[start:end]
        assert extracted_slice == c["exact_claim_text"], f"Offset mismatch: '{extracted_slice}' vs '{c['exact_claim_text']}'"
    print("PASS: 10. Character offsets strictly match original answer text slice!")

    print("\n=== ALL 10 ACCEPTANCE TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
