"""Hand-written labelled questions for Phase 2 (labels defined from passage text only, before
any retrieval system is evaluated). Outputs questions/dev.jsonl and questions/heldout.jsonl."""
import json
from pathlib import Path

# (qid, question, relevant passage ids, hard-negative passage ids, query types, reference answer)
DEV = [
 ("D01", "What is the nightly hotel limit in London under the current travel policy?", ["TRV24-03"], ["TRV22-03"], ["numerical", "distractor"], "260 euros per night (London is a designated high-cost city)."),
 ("D02", "How long do I have to file my expenses after I get back from a trip?", ["TRV24-06"], ["TRV22-05"], ["paraphrase", "numerical", "distractor"], "Within 30 days of returning; claims after 90 days are rejected."),
 ("D03", "Which cities count as high-cost for hotel bookings?", ["TRV24-03"], [], ["factual"], "London, Zurich, New York and Tokyo."),
 ("D04", "What mileage rate does Northwind pay for using my own car?", ["TRV24-05"], [], ["numerical"], "0.42 euros per kilometre, up to 400 km per trip."),
 ("D05", "Under the 2022 policy, what was the daily meal allowance?", ["TRV22-04"], ["TRV24-04"], ["numerical", "distractor"], "50 euros per day."),
 ("D06", "How did the flight duration threshold for business class change between the 2022 and 2024 travel policies?", ["TRV22-02", "TRV24-02"], [], ["multi_passage", "numerical"], "2022: business class from eight hours with Finance Director approval; 2024: only for flights over ten hours with CFO approval."),
 ("D07", "How quickly must a lost laptop be reported?", ["SEC-03"], [], ["factual", "numerical"], "Within one hour of discovery, to the security desk (extension 4400 or the incident form)."),
 ("D08", "Do I have to rotate my password every few months?", ["SEC-01"], [], ["paraphrase"], "No. Passwords do not expire periodically; change it immediately only if compromise is suspected."),
 ("D09", "Which accounts require hardware security keys?", ["SEC-02"], [], ["factual"], "Administrator accounts."),
 ("D10", "How many days' notice are required to stop the Contoso agreement from auto-renewing?", ["MSA-01"], ["FAB-01"], ["numerical", "distractor"], "At least 90 days before the end of the current term."),
 ("D11", "What is the early termination fee if we exit the Contoso contract for convenience?", ["MSA-02"], [], ["numerical"], "50 percent of the fees remaining for the current term, with 60 days written notice."),
 ("D12", "What remedy do we get if Contoso misses its uptime commitment?", ["MSA-03"], ["FAB-03"], ["factual", "distractor"], "Service credits of 10 percent of the monthly fee, 25 percent below 99.0 percent; credits are the sole remedy."),
 ("D13", "What is Fabrikam's liability cap?", ["FAB-04"], ["MSA-04"], ["numerical", "distractor"], "The fees paid in the 3 months before the claim."),
 ("D14", "Can Fabrikam use our routing data for its own purposes?", ["FAB-05"], [], ["paraphrase"], "Yes, in aggregated and anonymised form to improve its routing models."),
 ("D15", "Which vendor notifies customers of a data breach faster, and by how much?", ["MSA-05", "FAB-05"], [], ["multi_passage", "numerical"], "Contoso (48 hours) is 24 hours faster than Fabrikam (72 hours)."),
 ("D16", "What was the patient's haemoglobin in January 2025?", ["LABJ-01"], ["LABF-01"], ["numerical", "distractor"], "10.9 g/dL, below the 13.5 to 17.5 reference range."),
 ("D17", "Do the January results show signs of iron deficiency?", ["LABJ-02", "LABJ-05"], ["LABF-02"], ["paraphrase", "distractor"], "Yes: ferritin 9 ng/mL and transferrin saturation 11 percent; the pathologist suggests iron-deficiency anaemia."),
 ("D18", "By how much did serum ferritin rise between January and July?", ["LABJ-02", "LABF-02"], [], ["multi_passage", "numerical"], "From 9 to 38 ng/mL, a rise of 29 ng/mL."),
 ("D19", "What did the July pathologist recommend about glucose testing?", ["LABF-05"], ["LABJ-05"], ["factual", "distractor"], "Repeat glucose testing in twelve months."),
 ("D20", "Where does glycolysis happen and does it need oxygen?", ["RESP-02"], [], ["factual"], "In the cytoplasm; it does not need oxygen."),
 ("D21", "Where does the Calvin cycle take place?", ["PHOTO-03"], ["RESP-03"], ["factual", "distractor"], "In the stroma of the chloroplast."),
 ("D22", "How many ATP does oxidative phosphorylation produce?", ["RESP-04"], ["RESP-06"], ["numerical"], "About 34 ATP."),
 ("D23", "What happens to pyruvate in muscles when there is no oxygen?", ["RESP-05"], [], ["paraphrase"], "It is converted to lactate."),
 ("D24", "How are the overall equations of photosynthesis and cellular respiration related?", ["RESP-01", "PHOTO-05"], [], ["multi_passage"], "They are reverses of each other: respiration turns glucose and oxygen into carbon dioxide and water; photosynthesis does the opposite."),
]
HELDOUT = [
 ("H01", "What is the hotel cap for a standard city under the 2024 rules?", ["TRV24-03"], ["TRV22-03"], ["numerical", "distractor"], "180 euros per night."),
 ("H02", "Under the old travel policy, how long did employees have to submit claims?", ["TRV22-05"], ["TRV24-06"], ["numerical", "distractor"], "Within 60 days of the end of the trip."),
 ("H03", "Can I expense wine at a team dinner on a business trip?", ["TRV24-04"], [], ["paraphrase"], "No, alcohol is not reimbursable."),
 ("H04", "Who approves exceptions to the current travel policy?", ["TRV24-01"], ["TRV22-01"], ["factual", "distractor"], "The Chief Financial Officer, in writing."),
 ("H05", "For a seven-hour flight, which cabin can I book under the 2024 policy?", ["TRV24-02"], ["TRV22-02"], ["numerical", "distractor"], "Premium economy (flights of six hours or longer)."),
 ("H06", "Compare the daily meal allowance in the 2022 and 2024 policies.", ["TRV22-04", "TRV24-04"], [], ["multi_passage", "numerical"], "It rose from 50 euros (2022) to 65 euros (2024)."),
 ("H07", "What is the minimum password length?", ["SEC-01"], [], ["numerical"], "14 characters."),
 ("H08", "After how long does an idle company laptop lock?", ["SEC-04"], [], ["numerical"], "After five minutes of inactivity."),
 ("H09", "Can I copy payroll files to a USB stick?", ["SEC-05"], [], ["paraphrase"], "No. Payroll data is restricted and may not be stored on removable media."),
 ("H10", "What uptime does Contoso guarantee and what credit applies below 99 percent?", ["MSA-03"], ["FAB-03"], ["numerical", "distractor"], "99.9 percent monthly availability; below 99.0 percent the credit is 25 percent of the monthly fee."),
 ("H11", "Which country's law governs the Contoso agreement?", ["MSA-06"], [], ["factual"], "Ireland (courts of Dublin)."),
 ("H12", "How far ahead must we cancel the Fabrikam subscription to avoid renewal?", ["FAB-01"], ["MSA-01"], ["numerical", "distractor"], "At least 30 days before renewal, through the account portal."),
 ("H13", "By how much can Fabrikam raise prices at renewal?", ["FAB-02"], [], ["numerical"], "Up to 7 percent per year; above that the customer may terminate without penalty within 30 days."),
 ("H14", "Which contract has the higher availability commitment, and what are the two figures?", ["MSA-03", "FAB-03"], [], ["multi_passage", "numerical"], "Contoso: 99.9 percent versus Fabrikam: 99.5 percent."),
 ("H15", "Does the Contoso liability cap apply to indemnification claims?", ["MSA-04"], ["FAB-04"], ["factual", "distractor"], "No, indemnification obligations are excluded from the cap."),
 ("H16", "How did the patient's HbA1c change from January to July 2025?", ["LABJ-03", "LABF-03"], [], ["multi_passage", "numerical"], "It fell from 6.1 percent to 5.8 percent, still in the prediabetes range."),
 ("H17", "What was the LDL cholesterol in the follow-up report?", ["LABF-04"], ["LABJ-04"], ["numerical", "distractor"], "131 mg/dL."),
 ("H18", "Were the patient's red blood cells smaller than normal in January?", ["LABJ-01", "LABJ-05"], ["LABF-01"], ["paraphrase", "distractor"], "Yes, MCV was 72 fL (below 80); the pathologist describes microcytic anaemia."),
 ("H19", "What treatment did the patient receive between the two reports?", ["LABF-02"], [], ["factual"], "Three months of oral iron supplementation."),
 ("H20", "In which part of the mitochondrion does the Krebs cycle occur?", ["RESP-03"], ["PHOTO-03"], ["factual", "distractor"], "The mitochondrial matrix."),
 ("H21", "What temperature causes the rate of photosynthesis to fall sharply?", ["PHOTO-04"], [], ["numerical"], "Above about 40 degrees Celsius, when enzymes such as rubisco denature."),
 ("H22", "What by-product is released when water is split in the light reactions?", ["PHOTO-02"], [], ["paraphrase"], "Oxygen."),
 ("H23", "What do yeast cells produce during anaerobic respiration?", ["RESP-05"], [], ["factual"], "Ethanol and carbon dioxide."),
 ("H24", "Compare the total ATP yield of aerobic respiration with anaerobic respiration.", ["RESP-06", "RESP-05"], ["RESP-04"], ["multi_passage", "numerical"], "Aerobic: about 36 to 38 ATP per glucose; anaerobic: only the 2 ATP from glycolysis."),
]


def main():
    out = Path(__file__).parent / "questions"
    out.mkdir(exist_ok=True)
    for split, rows in (("dev", DEV), ("heldout", HELDOUT)):
        lines = [json.dumps({"qid": q, "split": split, "question": text, "relevant_passage_ids": rel,
                             "hard_negative_passage_ids": neg, "query_types": types, "reference_answer": ans},
                            ensure_ascii=False) for q, text, rel, neg, types, ans in rows]
        (out / f"{split}.jsonl").write_text("\n".join(lines) + "\n")
    print(len(DEV), "dev /", len(HELDOUT), "held-out")


if __name__ == "__main__":
    main()
