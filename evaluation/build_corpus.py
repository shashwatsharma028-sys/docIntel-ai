"""Source of truth for the synthetic Phase 2 evaluation corpus (all content fictitious).
Run once to (re)generate corpus/passages.jsonl and corpus/<doc>.txt; the outputs are frozen
by LABELS.lock.json and must not be regenerated after labels are frozen."""
import json
from pathlib import Path

DOCS = {
 "northwind_travel_policy_2024": ("Northwind Logistics Travel and Expense Policy (2024 edition, effective 1 April 2024)", "enterprise", [
  ("TRV24-01", "Scope. This policy applies to all Northwind employees and contractors travelling on company business from 1 April 2024. It replaces the 2022 travel policy in full. Exceptions require written approval from the Chief Financial Officer."),
  ("TRV24-02", "Air travel. Economy class is required for flights under six hours. Premium economy may be booked for flights of six hours or longer. Business class requires CFO approval and is limited to flights over ten hours."),
  ("TRV24-03", "Hotels. The nightly room limit is 180 euros in standard cities and 260 euros in designated high-cost cities, which are London, Zurich, New York and Tokyo. Rates above the limit need line manager approval before booking."),
  ("TRV24-04", "Meals. Employees may claim a daily meal allowance of 65 euros on travel days. Alcohol is not reimbursable. Client entertainment is claimed separately under the hospitality procedure with the names of all attendees."),
  ("TRV24-05", "Mileage. Use of a private car is reimbursed at 0.42 euros per kilometre up to 400 kilometres per trip. Longer journeys must use rail unless the line manager approves the car in advance."),
  ("TRV24-06", "Claims. Expense claims must be submitted in the finance portal within 30 days of returning, with itemised receipts for every item above 25 euros. Claims submitted after 90 days are rejected."),
 ]),
 "northwind_travel_policy_2022": ("Northwind Logistics Travel Policy (2022 edition, superseded)", "enterprise", [
  ("TRV22-01", "Scope. This 2022 policy applies to employees travelling on company business. Contractors are covered by their own agreements. Exceptions require approval from the Finance Director."),
  ("TRV22-02", "Air travel. Economy class is required for flights under eight hours. Business class may be booked for flights of eight hours or longer with Finance Director approval."),
  ("TRV22-03", "Hotels. The nightly room limit is 150 euros in all cities. Employees should prefer hotels on the corporate rate list."),
  ("TRV22-04", "Meals. The daily meal allowance is 50 euros. Receipts are required for meals above 15 euros."),
  ("TRV22-05", "Claims. Expense claims must be submitted within 60 days of the end of the trip. Late claims may be paid at the discretion of the Finance Director."),
 ]),
 "northwind_it_security_policy": ("Northwind Logistics Information Security Policy IS-07", "enterprise", [
  ("SEC-01", "Passwords. Passwords must be at least 14 characters long and are not subject to periodic expiry. A password must be changed immediately if compromise is suspected. Password reuse across company systems is prohibited."),
  ("SEC-02", "Multi-factor authentication. MFA is mandatory for email, VPN, the finance portal and all administrator accounts. Hardware security keys are required for administrators; other staff may use the authenticator app."),
  ("SEC-03", "Incident reporting. Suspected security incidents, including lost laptops and phishing clicks, must be reported to the security desk within one hour of discovery by phone on extension 4400 or through the incident form."),
  ("SEC-04", "Devices. Company laptops use full-disk encryption and lock automatically after five minutes of inactivity. Personal devices may access email only through the managed mobile application."),
  ("SEC-05", "Data classification. Information is classified as public, internal, confidential or restricted. Restricted data, such as payroll and medical records, may not be stored on removable media or sent to personal email addresses."),
 ]),
 "contoso_cloud_msa": ("Master Services Agreement between Contoso Cloud Hosting Ltd and Customer", "legal", [
  ("MSA-01", "Term and renewal. The initial term is 36 months. The agreement renews automatically for successive 12-month periods unless either party gives written notice of non-renewal at least 90 days before the end of the current term."),
  ("MSA-02", "Termination for convenience. The customer may terminate for convenience on 60 days written notice by paying an early termination fee equal to 50 percent of the fees remaining for the current term."),
  ("MSA-03", "Service levels. Contoso commits to 99.9 percent monthly availability. If availability falls below that level the customer receives a service credit of 10 percent of the monthly fee, rising to 25 percent below 99.0 percent. Service credits are the sole remedy for downtime."),
  ("MSA-04", "Limitation of liability. Each party's total liability is capped at the fees paid in the 12 months before the claim. The cap does not apply to breach of confidentiality, the customer's payment obligations or indemnification obligations."),
  ("MSA-05", "Data handling. Contoso stores customer data only in data centres located in the European Economic Area and will notify the customer of a personal data breach within 48 hours of becoming aware of it."),
  ("MSA-06", "Governing law. This agreement is governed by the laws of Ireland. Disputes are resolved by the courts of Dublin, and each party waives the right to a jury trial."),
 ]),
 "fabrikam_logistics_saas": ("Fabrikam Route Optimisation SaaS Subscription Terms", "legal", [
  ("FAB-01", "Term and renewal. The subscription runs for 12 months and renews automatically for further 12-month terms unless the customer cancels at least 30 days before renewal through the account portal."),
  ("FAB-02", "Fees and price changes. Fabrikam may increase subscription fees at renewal by up to 7 percent per year. Price increases above 7 percent allow the customer to terminate without penalty within 30 days of notice."),
  ("FAB-03", "Service levels. Fabrikam targets 99.5 percent monthly availability. Service credits of 5 percent of the monthly fee apply below the target and must be requested within 15 days of the end of the month."),
  ("FAB-04", "Liability. Fabrikam's liability is capped at the fees paid in the 3 months before the claim, and Fabrikam is not liable for indirect or consequential losses, including lost profits."),
  ("FAB-05", "Data. Customer route data may be used by Fabrikam in aggregated and anonymised form to improve its routing models. Fabrikam will notify customers of a data breach within 72 hours."),
 ]),
 "riverside_lab_report_jan2025": ("Riverside Diagnostics laboratory report, patient R-1047, collected 14 January 2025", "medical", [
  ("LABJ-01", "Full blood count. Haemoglobin 10.9 g/dL (reference 13.5 to 17.5), mean corpuscular volume 72 fL (reference 80 to 100). White cell count 7.2 and platelets 265, both within normal limits."),
  ("LABJ-02", "Iron studies. Serum ferritin 9 ng/mL (reference 30 to 400) and transferrin saturation 11 percent (reference 20 to 50), consistent with iron deficiency."),
  ("LABJ-03", "Glucose metabolism. Fasting glucose 118 mg/dL (normal below 100; 100 to 125 indicates impaired fasting glucose). HbA1c 6.1 percent (5.7 to 6.4 indicates prediabetes)."),
  ("LABJ-04", "Lipids. Total cholesterol 245 mg/dL, LDL cholesterol 162 mg/dL (optimal below 100), HDL 41 mg/dL and triglycerides 210 mg/dL."),
  ("LABJ-05", "Pathologist comment. Findings suggest microcytic anaemia due to iron deficiency and impaired fasting glucose. Recommend clinical correlation, investigation of the cause of iron loss and repeat testing in six months."),
 ]),
 "riverside_lab_report_jul2025": ("Riverside Diagnostics laboratory report, patient R-1047, collected 16 July 2025 (follow-up)", "medical", [
  ("LABF-01", "Full blood count. Haemoglobin 12.8 g/dL (reference 13.5 to 17.5), mean corpuscular volume 81 fL. White cell count 6.5 and platelets 240, within normal limits."),
  ("LABF-02", "Iron studies. Serum ferritin 38 ng/mL (reference 30 to 400) after three months of oral iron supplementation; transferrin saturation 22 percent."),
  ("LABF-03", "Glucose metabolism. Fasting glucose 104 mg/dL and HbA1c 5.8 percent, both still in the prediabetes range but lower than in January."),
  ("LABF-04", "Lipids. LDL cholesterol 131 mg/dL and triglycerides 160 mg/dL after dietary changes; HDL 45 mg/dL."),
  ("LABF-05", "Pathologist comment. Marked improvement in iron stores. Haemoglobin is approaching the reference range. Continue supplementation and review with the treating physician; repeat glucose testing in twelve months."),
 ]),
 "biology_cellular_respiration_notes": ("Biology Chapter Four Study Notes: Cellular Respiration", "study", [
  ("RESP-01", "Overview. Cellular respiration releases energy from glucose to make ATP. The overall equation is glucose plus six oxygen yields six carbon dioxide plus six water plus energy, the reverse of the photosynthesis equation."),
  ("RESP-02", "Glycolysis takes place in the cytoplasm, does not need oxygen and splits one glucose into two pyruvate molecules, giving a net gain of two ATP and two NADH."),
  ("RESP-03", "The Krebs cycle, also called the citric acid cycle, takes place in the mitochondrial matrix. Each turn releases two carbon dioxide molecules and produces NADH, FADH2 and one ATP."),
  ("RESP-04", "Oxidative phosphorylation happens on the inner mitochondrial membrane. The electron transport chain pumps protons, and ATP synthase uses the proton gradient to produce about 34 ATP. Oxygen is the final electron acceptor and forms water."),
  ("RESP-05", "Anaerobic respiration. Without oxygen, human muscle cells convert pyruvate to lactate, while yeast produces ethanol and carbon dioxide. Anaerobic pathways yield only the two ATP from glycolysis."),
  ("RESP-06", "Exam summary. Aerobic respiration yields about 36 to 38 ATP per glucose in total. Key terms: glycolysis, pyruvate, Krebs cycle, mitochondrial matrix, electron transport chain, ATP synthase."),
 ]),
 "biology_photosynthesis_notes": ("Biology Chapter Three Study Notes: Photosynthesis", "study", [
  ("PHOTO-01", "Overview. Photosynthesis converts light energy into chemical energy stored in glucose. It takes place in chloroplasts, which contain the green pigment chlorophyll in the thylakoid membranes."),
  ("PHOTO-02", "The light-dependent reactions occur in the thylakoid membranes. Photolysis splits water, releasing oxygen as a by-product and producing ATP and NADPH."),
  ("PHOTO-03", "The Calvin cycle takes place in the stroma. The enzyme rubisco fixes carbon dioxide, and ATP and NADPH from the light reactions are used to make three-carbon sugars that build glucose."),
  ("PHOTO-04", "Limiting factors. The rate of photosynthesis is limited by light intensity, carbon dioxide concentration and temperature. Above about 40 degrees Celsius enzymes such as rubisco denature and the rate falls sharply."),
  ("PHOTO-05", "Exam summary. Overall equation: six carbon dioxide plus six water yields glucose plus six oxygen. Key terms: chlorophyll, thylakoid, stroma, photolysis, Calvin cycle, rubisco."),
 ]),
}


def main():
    out = Path(__file__).parent / "corpus"
    out.mkdir(exist_ok=True)
    rows = []
    for doc_id, (title, domain, passages) in DOCS.items():
        (out / f"{doc_id}.txt").write_text(title + "\n\n" + "\n\n".join(t for _, t in passages) + "\n")
        for order, (pid, text) in enumerate(passages):
            rows.append({"passage_id": pid, "doc_id": doc_id, "source": f"{doc_id}.pdf", "domain": domain,
                         "title": title, "order": order, "text": text})
    ids = [r["passage_id"] for r in rows]
    assert len(ids) == len(set(ids)), "passage ids must be unique"
    (out / "passages.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(DOCS)} documents, {len(rows)} passages")


if __name__ == "__main__":
    main()
