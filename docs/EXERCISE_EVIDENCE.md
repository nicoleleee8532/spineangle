# Exercise evidence by curve pattern

Research notes behind the SpineAngle exercise panel. November 2026.

> The tool explains what physiotherapy for each curve pattern aims to do. It does not prescribe exercises, sets or reps. Every source below delivers exercise through a trained physiotherapist who assesses the person first.

## 1. What "scoliosis-specific exercise" means

Physiotherapeutic scoliosis-specific exercises (PSSE) are different from general exercise. A comprehensive review of the seven major schools — Lyon (France), Schroth (Germany), SEAS (Italy), Barcelona/BSPTS (Spain), Dobomed (Poland), Side Shift (UK) and FITS (Poland) — found they share the same core components:

- 3D self-correction of the spine
- training daily activities while holding the corrected posture
- stabilising that corrected posture
- use within a wider care model (education, observation, bracing and surgery when indicated), run by a multidisciplinary team

The review reports benefits beyond the Cobb angle: improved trunk asymmetry, reduced muscle imbalance and pain, and in severe thoracic curves, better breathing. [Berdishevsky et al., 2016, PubMed 27525315](https://pubmed.ncbi.nlm.nih.gov/27525315/)

The 2016 SOSORT international guidelines list the same essential elements: 3D self-correction, training in daily activities, stabilisation of the corrected posture, and patient education. For progressive curves over 25° during growth they recommend bracing, and say PSSE alone without a brace should only be used when prescribed by a scoliosis-expert physician. [SOSORT 2016 guidelines](https://link.springer.com/10.1186/s13013-017-0145-8)

## 2. Trial evidence

| Study | Who | What | Result |
|-------|-----|------|--------|
| Kuru et al., 2016, RCT | 45 adolescents with AIS | Schroth supervised in clinic vs the same exercises at home vs no treatment, 24 weeks | Clinic group: Cobb angle down about 2.5°, rotation and rib hump improved, waist asymmetry improved only in this group. Home exercise did not match supervised. No change in quality of life. Untreated curves progressed. [Clin Rehabil](https://journals.sagepub.com/doi/10.1177/0269215515575745) |
| Monticone et al., 2014, RCT | 110 adolescents, Cobb 10–25° | Active self-correction + task-oriented exercise + education vs general exercise, weekly supervised + home, until skeletal maturity | Cobb −5.3° vs +1.7° in controls; less pain; effect held at 12 months; minor transient pain flare-ups in both groups. [Trial PDF](https://re.public.polimi.it/bitstream/11311/865753/4/Active%20self-correction%20and%20task-oriented%20exercises%20reduce%20spinal%20deformity_11311-865753_Ambrosini.pdf) |
| Monticone et al., 2016, RCT | Adults with idiopathic scoliosis | Motor and cognitive rehabilitation vs general physiotherapy | Reported reduced disability (title of the trial); PEDro quality score 7/10. I could not read the abstract, so effect sizes aren't stated here. [Eur Spine J record](https://search.pedro.org.au/search-results/record-detail/45636) |

Other evidence:

- A 2024 Cochrane review, *Therapeutic exercises for idiopathic scoliosis in adolescents*, updates the 2012 version. I couldn't access its results, so it isn't summarised here; read it before quoting any effect size. [Cochrane CD007837.pub3](https://cochranelibrary.com/cdsr/doi/10.1002/14651858.CD007837.pub3/information)
- For adults, a 2019 survey of spine surgeons found considerable scepticism about PSSE, and the authors describe adult evidence as largely limited to case reports and case series. [Int J Spine Surg](https://www.ijssurgery.com/content/ijss/13/6/568.full.pdf)

**Takeaways for the tool**

1. Supervision matters: home exercise without a physio did not reproduce the supervised effect.
2. Evidence is mostly in adolescents with mild to moderate curves. For adults, it supports reduced disability and pain more than reducing the angle.
3. Exercises look low-risk (only minor, transient pain flare-ups reported).

## 3. Why the tool can't pick a Schroth exercise type from an X-ray

The Schroth method classifies curves into four types:

- **3c**: major thoracic curve, small or no lumbar curve, balanced pelvis
- **3cp**: long thoracic curve, pelvis shifted opposite the thoracic convexity
- **4c**: major lumbar curve with a similar thoracic curve, balanced pelvis
- **4cp**: major thoracolumbar or lumbar curve, pelvis shifted opposite the lumbar convexity

The classification is made by **clinical visual assessment** of the hip-pelvis, lumbar, thoracic and shoulder blocks, not from the X-ray, and the exercises differ by type. Even trained therapists only reached moderate agreement with each other (inter-rater AC1 0.43 across all raters, 0.67 for the most experienced). [Schroth classification reliability study, 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10047876)

So an X-ray angle can tell you the **radiographic pattern** (where the curves are and which way they bend) but not the pelvis balance that decides the Schroth type. That's the honest limit of the tool, and it's why the panel prepares the user for a physio assessment instead of replacing it.

## 4. What the tool does

1. **Classifies the radiographic pattern** from the measured curves:
   - curve region by apex position: thoracic, thoracolumbar or lumbar
   - pattern: single thoracic, single thoracolumbar/lumbar, or double (S-shaped)
   - which side each curve bends toward on the image
2. **Explains what physio typically focuses on** for that pattern, at the level of the shared PSSE components above, with links.
3. **Gives questions to ask a physio**, so the person gets the most from free NHS sessions.
4. **States safety rules**: supervised first, stop if pain worsens, red-flag symptoms.

## 5. Where the machine learning fits

There is no public dataset linking curve patterns to exercise outcomes, so an exercise-recommendation model would have no valid labels. Instead the ML is in pattern classification (`ml/pattern.py`):

- **Input:** only what manual mode knows: the marked endplate points and their position in the cropped image, without vertebra labels.
- **Target:** the true curve region, computed from ground-truth vertebra levels in AASCE landmarks.
- **Model:** softmax regression in numpy, exported to JSON so the browser can run it.

It currently trains on synthetic spines; retrain on AASCE once the data is available, and report accuracy against the hand-written thresholds the browser uses today.

## 6. Side plank

| Study | Who | What | Result |
|-------|-----|------|--------|
| Fishman et al., 2014, retrospective case series | 25 people aged 14–85, idiopathic and degenerative scoliosis | Side plank held once daily, as long as possible, on the convex side of the primary curve (convex side down); 10–20 s a day in week 1. Average 1.5 min a day, about 6 days a week | Primary curve 37.2° → 25.3° (32%) at a mean 6.8 months; 41% in compliant patients vs 0.5% in non-compliant. No control group; only side effect mild wrist or shoulder discomfort. [PMC4268609](https://pmc.ncbi.nlm.nih.gov/articles/PMC4268609/) |
| Sarkisova et al., 2019, RCT | 64 adolescents, curves 10–45°; only 22 compliant | Side plank (convex down) vs front plank, 6 months | No significant change in Cobb angle in either group. [PubMed 31723480](https://pubmed.ncbi.nlm.nih.gov/31723480/) |
| Fishman, 2021, letter | – | Critique of the RCT | Says the pose is meant only for lumbar and thoracolumbar curves and can worsen thoracic curves; the RCT's intervention group had almost no lumbar curves and used a different foot position. [Global Adv Health Med](https://journals.sagepub.com/doi/reader/10.1177/21649561211019372) |

**In the tool:** for a lumbar or thoracolumbar major curve, the side plank is shown with the bulging side down, the study's build-up, and the mixed evidence. For a thoracic-only pattern it is shown as "check with a physio first", following its creator's own caution.

## 7. Videos

Linked (the viewer can't embed YouTube), chosen from hospitals, clinics and therapists:

- Scottish Rite for Children: how Schroth therapy works
- The ScoliClinic (physio Andrea): five basic Schroth positions; a side-lying Schroth demo
- Dr Katie Blanchard (Schroth Boss): home exercises for a thoracic-right, lumbar-left curve (shown for double curves)
- Move With Scoliosis (yoga and Pilates teacher) with Dr Loren Fishman; and which side to do the side plank

None are tailored to an individual curve, and the tool says so.

## 8. Personal plan

**Mirror check.** The user answers: which side of the upper back is higher when bending forward (rib hump, the thoracic convex side), which side of the waist creases more (the lumbar concave side), whether the hips look shifted, plank ability, pain frequency and whether they are still growing.

- The hump or crease answer maps the image's sides to the body's, so every direction is given as "your left" or "your right".
- With a pelvis answer, the tool estimates a Schroth type from the definitions in section 3: level pelvis gives 3c (thoracic main) or 4c (lumbar main); a pelvis shifted opposite the main curve's convexity gives 3cp or 4cp, which adds a pelvis step to self-correction. Other combinations are left to the physio. Given the moderate agreement between trained therapists, this is labelled an estimate.
- Frequent pain or not managing a 30-second plank switches to the easier versions. Still growing adds a prompt to see a spine specialist about bracing.

**Exercises shown**, each with a drawing mirrored to the user's sides:

| Exercise | Shown for | Why |
|----------|-----------|-----|
| Self-correction in standing | all | Shared core of every PSSE school (section 1) |
| Breathing into the hollow, lying, towel under the bulge side | thoracic and double | Schroth rotational breathing: inhale into the concave side, with a small bag or wedge under the convex side ([Acibadem overview](https://acibademinternational.com/blog/what-happens-in-a-schroth-method-session-elongation-rotational-breathing-and-corrections/)) |
| Side plank, bulge side down | lumbar, thoracolumbar and double | Section 6; withheld for thoracic-only patterns |
| Front plank, bird-dog, glute bridge | all | General trunk and hip strength. In a trial of 28 teenagers with mild curves, Schroth improved the Cobb angle, rotation and quality of life more than core stabilisation, while core stabilisation improved muscle strength more ([Kocaman et al., 2021](https://doaj.org/article/9b92f055635f4d6bb59f7c277bb52704)), so these are framed as strength support, not curve correction |

Doses for the strength exercises are conservative beginner amounts (short holds, few reps), not values from scoliosis trials; the side plank build-up follows Fishman 2014.
