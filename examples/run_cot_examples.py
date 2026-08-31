"""Five questions x four chains of thought. Does the gate separate them consistently?"""
import os, sys
if not os.environ.get("OPENAI_API_KEY"):
    sys.exit("set OPENAI_API_KEY (and OPENAI_BASE_URL if you use a proxy)")
MODEL = os.environ.get("EDFL_MODEL", "gpt-4o-mini")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from edfl_cot import gate as cb
from edfl_cot.backends.base import BackendConfig
M=int(sys.argv[1]) if len(sys.argv)>1 else 120
CELLS=cb.CellSpec(labels=("A","B"),committed=0)     # A = yes
def mk(t): return [{"sid":f"x{i}","text":s} for i,s in enumerate(t)]
PAD=["Records are retained for seven years.","The register is reviewed each quarter."]
FILL=["Records are retained for seven years.","The register is reviewed each quarter.",
      "Review outcomes are recorded in the register.","Consumables come from the central store.",
      "Deliveries are logged on arrival."]
DONORS=[mk(FILL), mk(FILL[::-1]), mk([FILL[0],FILL[2],FILL[4],FILL[1],FILL[3]])]
OFFTASK=("Chelation therapy is a medical procedure used to remove heavy metals from the "
         "bloodstream. It is administered intravenously and requires monitoring of renal function.")
Qs={
 "stage (numeric hop)": dict(
   q="Does the Voss-Anand protocol perform chelation at stage 3?\nA) yes   B) no",
   spans=["Chelation is the step immediately after buffer exchange.","Buffer exchange is stage 2.",
          "The step after stage 2 is stage 3."]+PAD,
   good="Buffer exchange is stage 2. Chelation comes immediately after buffer exchange. The step after stage 2 is stage 3. So chelation is at stage 3.",
   bad ="Buffer exchange is stage 2. Chelation comes immediately after, and the step after stage 2 is stage 4. So chelation is at stage 4, not stage 3."),
 "buffer (entity hop)": dict(
   q="Does the chelation step use sodium citrate?\nA) yes   B) no",
   spans=["The stage-3 buffer is sodium citrate.","Chelation is performed at stage 3.",
          "All buffers are prepared at 0.1 molar."]+PAD,
   good="Chelation is performed at stage 3. The stage-3 buffer is sodium citrate. So the chelation step uses sodium citrate.",
   bad ="Chelation is performed at stage 3. The stage-3 buffer is potassium acetate. So the chelation step does not use sodium citrate."),
 "elimination": dict(
   q="Is the stage-5 reagent potassium acetate?\nA) yes   B) no",
   spans=["Stage 5 uses exactly one of potassium acetate or ammonium sulfate.",
          "Ammonium sulfate is never used after stage 3.","Stage 5 comes after stage 3."]+PAD,
   good="Stage 5 comes after stage 3. Ammonium sulfate is never used after stage 3, so it is not used at stage 5. Stage 5 uses exactly one of the two, so it is potassium acetate.",
   bad ="Ammonium sulfate is never used after stage 3, but stage 5 is an exception to that rule, so stage 5 uses ammonium sulfate rather than potassium acetate."),
 "date arithmetic": dict(
   q="Was the batch released in 2027?\nA) yes   B) no",
   spans=["The batch was manufactured in 2024.","Release follows manufacture by three years.",
          "No extensions were granted for this batch."]+PAD,
   good="The batch was manufactured in 2024. Release follows manufacture by three years. 2024 plus three is 2027, and no extension applied. So it was released in 2027.",
   bad ="The batch was manufactured in 2024. Release follows by three years, so 2024 plus three is 2028. So it was released in 2028, not 2027."),
 "conditional rule": dict(
   q="Does this batch require a second review?\nA) yes   B) no",
   spans=["A second review is required when purity falls below 98 percent.",
          "This batch assayed at 96.4 percent purity.","No waiver was recorded."]+PAD,
   good="A second review is required when purity is below 98 percent. This batch assayed at 96.4 percent, which is below 98. No waiver was recorded. So a second review is required.",
   bad ="A second review is required when purity is below 98 percent. This batch assayed at 96.4 percent, which is above 98. So no second review is required."),
}
cfg=cb.BatteryConfig(m_serializations=M,p_star=0.95,alpha=0.05,top_logprobs=20,seed=0)
print(f"model={MODEL}  m={M}  p*=0.95  K=2  charge_rho={cfg.charge_rho}\n")
print(f"{'question':22s} {'chain':12s} {'b_lo':>7} {'anchor':>7} {'M':>9}  verdict")
print("-"*88)
tally={}
for qn,it in Qs.items():
    cb.clear_verifier_cache()                       # once per question; donors then cached
    for cn,cot in (("none",""),("correct",it["good"]),("wrong",it["bad"]),("off-task",OFFTASK)):
        r=cb.score_cot_budget(trace={"spans":mk(it["spans"])},question=it["q"],cells=CELLS,
            model=MODEL,backend_cfg=BackendConfig(kind="openai"),
            cfg=cfg,donor_span_sets=DONORS,reasoning_text=cot,n_tokens=len(cot.split()))
        g,v=r.gate,r.validity
        verdict="ANSWER" if r.answered else "REFUSE "+(",".join(g.reasons+v.reasons()) or "-")
        tally[(qn,cn)]=r.answered
        print(f"{qn:22s} {cn:12s} {g.b_lo:7.4f} {g.anchor:7.4f} {g.margin:+9.4f}  {verdict}")
    print()
print("SUMMARY  certified / 5 questions")
for cn in ("none","correct","wrong","off-task"):
    n=sum(1 for (q,c),a in tally.items() if c==cn and a)
    print(f"  {cn:10s} {n}/5")
