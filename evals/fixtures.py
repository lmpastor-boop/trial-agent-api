"""
The exact 8-case hand-labeled test set and 3 real, frozen AML trials from
clinical_trial_agent_colab_REAL_API.ipynb (Section 14) -- copied verbatim so
the meta-eval is checked against the SAME ground truth used for the capstone's
validated 7/8 (88%) result, not a re-typed (and potentially drifted) copy.

Frozen as of 2026-07-16, same as the notebook.
"""

REAL_TRIALS = {
    "NCT05886049": {
        "nct_id": "NCT05886049",
        "title": "A Phase 1b Study of Menin Inhibitor SNDX-5613 in Combination With Daunorubicin and Cytarabine in Newly Diagnosed Patients With Acute Myeloid Leukemia and NPM1 Mutated/FLT3 Wildtype or MLL/KMT2A Rearranged or NUP98 Alterations Disease",
        "eligibility_text": """Inclusion Criteria:

* Dose escalation: Patients ages 18-75 years at time of diagnosis with NPM1-mutated/FLT3-ITD wildtype and NPM1-mutated/FLT3-TKD wildtype with high-risk features (adverse risk genetics per European LeukemiaNet [ELN] 2022 criteria, age >= 60 years, or secondary AML defined as either arising from a prior hematological malignancy or therapy-related), MLL (KMT2A) rearranged or NUP98 altered, untreated AML and who are candidates for intensive induction chemotherapy. Patients with CD33+ AML are eligible for this protocol.
* Dose expansion: Patients ages 18-75 years at time of diagnosis with NPM1-mutated/FLT3-ITD wildtype and NPM1-mutated/FLT3-TKD wildtype (any patient-does not require high-risk features), MLL (KMT2A) rearranged, or NUP98 altered, untreated AML and who are candidates for intensive induction chemotherapy. Patients with CD33+ AML are eligible for this protocol
* Because no dosing or adverse event data are currently available on the use of SNDX-5613 in combination with daunorubicin and cytarabine in patients < 18 years of age, children are excluded from this study
* Eastern Cooperative Oncology Group (ECOG) performance status =< 2 (Karnofsky >= 60%). Patients over the age of 65 must have an ECOG performance status of 0-1
* Total bilirubin <= 1.5 x institutional upper limit of normal (ULN), except for patients with Gilbert's syndrome where required to be <= 3 x institutional ULN
* AST/ALT =< 3 x institutional upper limit of normal (ULN)
* GFR >= 60 mL/min/1.73 m^2
* Patients must have previously untreated AML with no prior treatment other than hydroxyurea or intrathecal chemotherapy for CNS prophylaxis/treatment. No chemotherapy for AML outside of hydroxyurea for treatment of leukostasis or all-trans retinoic acid (ATRA) for initially suspected acute promyelocytic leukemia (APL) (that is ruled out) is allowed

Exclusion Criteria:

* Acute promyelocytic leukemia (French-American-British [FAB] M3)
* Patients with Down Syndrome due to higher rates of chemotherapy-associated toxicities, and may have different pharmacokinetics, as well
* Pregnant women are excluded from this study
* Patients with myelodysplastic syndromes (MDS) treated with previous intensive induction regimens similar to 7+3
(criteria trimmed for brevity in this test set -- full text pulled live via API)""",
        "min_age": 18,
        "max_age": 75,
    },
    "NCT05101551": {
        "nct_id": "NCT05101551",
        "title": "Study of Talazoparib in Combination With Chemotherapy in Relapsed Pediatric AML to Determine Safety and Efficacy",
        "eligibility_text": """Inclusion Criteria:

1. Aged <= 21 years.
2. Acute myeloid leukemia (AML) OR acute leukemia of ambiguous lineage, specified as either refractory (persistent leukemia after at least 2 courses of induction chemotherapy) or relapsed, and further defined by bone marrow/MRD criteria assessed by flow cytometry.
3. > 60 days has passed since hematopoietic stem cell transplant, if applicable, without active GVHD.
4. Lansky (subjects <= 16 years old) or Karnofsky (subjects > 16 years old) score >= 50.
5. WBC <= 50,000/uL.
6. Total bilirubin <= 2.0 x institutional ULN for age. AST/ALT <= 5 x ULN for age. LVEF >= 40%.

Exclusion Criteria:

1. Patients receiving or planning to receive ANY concurrent cancer therapy.
2. Patients with Down syndrome.
3. Patients with Acute Promyelocytic leukemia (APL) or Juvenile Myelomonocytic Leukemia (JMML).
4. Patients with Bone Marrow Failure Syndrome.
5. Pregnant subjects or those unwilling to use an effective method of birth control.
(criteria trimmed for brevity in this test set -- full text pulled live via API)""",
        "min_age": 0,
        "max_age": 21,
    },
    "NCT05092451": {
        "nct_id": "NCT05092451",
        "title": "Phase I/II Study of CAR.70-Engineered IL15-transduced Cord Blood-derived NK Cells for Relapse/Refractory Hematological Malignancies",
        "eligibility_text": """Inclusion criteria:

1. Patients with hematological malignancies with an expression of CD70 in the pre-enrollment tumor sample >= 10% measured by immunohistochemistry or flow cytometry.
2. Patients must meet disease-specific eligibility criteria.
3. At least 1 week from last cytotoxic chemotherapy at the time of starting lymphodepleting chemotherapy (hydroxyurea and select targeted therapies exempted).
4. Karnofsky Performance Scale > 50% (>16 years old) or Lansky score >=50% (<=16 years old).
5. Adequate renal, hepatic, cardiac, and pulmonary function per protocol-defined thresholds.
6. 12-80 years of age. Weight >= 40 kg.

Exclusion criteria:

1. Positive beta HCG / pregnancy.
2. Grade 3+ toxicity from previous treatment.
3. Uncontrolled active infection.
4. HIV with detectable viral load.
5. Active autoimmune disease within 12 months, or active (therapy-requiring) GVHD.
6. Any other active malignancy except treated cervical intraepithelial neoplasia or non-melanoma skin cancer.
(criteria trimmed for brevity in this test set -- full text pulled live via API)""",
        "min_age": 12,
        "max_age": 80,
    },
}

TEST_CASES = [
    {
        "patient": "Maria, 45F. Newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype. ECOG 1. Normal organ function labs. No prior treatment except hydroxyurea for WBC control. Candidate for intensive induction chemotherapy. CD70 expression: not tested.",
        "trial": "NCT05886049",
        "ground_truth": "Likely eligible",
        "why": "Matches the dose-expansion cohort directly: NPM1-mutated/FLT3-ITD wildtype, untreated AML, candidate for induction chemo, ECOG within range. No exclusion criteria triggered.",
    },
    {
        "patient": "Maria, 45F. Newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype. ECOG 1. Normal organ function labs. No prior treatment except hydroxyurea for WBC control. Candidate for intensive induction chemotherapy. CD70 expression: not tested.",
        "trial": "NCT05101551",
        "ground_truth": "Likely not eligible",
        "why": "Trial requires refractory/relapsed disease after >=2 prior induction courses, and age <=21. Maria is newly diagnosed and 45 -- both disqualify independently.",
    },
    {
        "patient": "David, 16M. Relapsed AML, 6% blasts by flow after 2 prior induction courses. No prior HSCT. Karnofsky/Lansky adequate. WBC 12,000. No Down syndrome. Labs within range.",
        "trial": "NCT05886049",
        "ground_truth": "Likely not eligible",
        "why": "Trial requires previously untreated, newly diagnosed AML. David has relapsed disease after induction chemo -- explicitly disqualifying regardless of age (age 16 also fails the 18 floor).",
    },
    {
        "patient": "David, 16M. Relapsed AML, 6% blasts by flow after 2 prior induction courses. No prior HSCT. Karnofsky/Lansky adequate. WBC 12,000. No Down syndrome. Labs within range.",
        "trial": "NCT05101551",
        "ground_truth": "Likely eligible",
        "why": "Age 16<=21, relapsed AML meeting the blast-by-flow criterion, no Down syndrome, WBC<=50,000, no transplant complication.",
    },
    {
        "patient": "Angela, 30F. Active Acute Promyelocytic Leukemia (APL), confirmed, untreated. ECOG 1. Otherwise healthy, normal labs.",
        "trial": "NCT05886049",
        "ground_truth": "Likely not eligible",
        "why": "Trial explicitly excludes APL (FAB M3). Pure free-text exclusion -- age/sex checks alone would never catch this. Tests whether the LLM actually reads the exclusion list.",
    },
    {
        "patient": "Tom, 22M. Newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype. Has Down syndrome. ECOG 1. Labs within range.",
        "trial": "NCT05886049",
        "ground_truth": "Likely not eligible",
        "why": "Trial explicitly excludes Down syndrome. Otherwise Tom matches the core inclusion criteria closely -- best case for catching an LLM that pattern-matches on diagnosis and skips exclusions.",
    },
    {
        "patient": "Sam, 60M. Secondary AML arising from prior MDS (therapy-related), NPM1-mutated/FLT3-ITD wildtype. Karnofsky 55%. Otherwise untreated, candidate for induction chemo.",
        "trial": "NCT05886049",
        "ground_truth": "Likely not eligible",
        "why": "Secondary AML from prior MDS IS an explicitly qualifying high-risk feature -- that part matches. But ECOG<=2/Karnofsky>=60% is required, and Sam's 55% falls below that. Hardest case in the set: the eligibility text bundles a qualifying feature and a disqualifying number in the same paragraph. Flag as a genuine judgment call if the LLM disagrees.",
    },
    {
        "patient": "Maria, 45F. Newly diagnosed AML, NPM1-mutated, FLT3-ITD wildtype. ECOG 1. Normal organ function labs. No prior treatment except hydroxyurea for WBC control. Candidate for intensive induction chemotherapy. CD70 expression: not tested.",
        "trial": "NCT05092451",
        "ground_truth": "Possibly eligible (needs more info)",
        "why": "Age 45 is within range and AML is plausible, but the trial requires CD70 expression >=10% as a hard inclusion criterion, and that's untested. Correct behavior is flagging the missing biomarker, not guessing. Tests whether the LLM actually uses the needs-more-info category.",
    },
]
