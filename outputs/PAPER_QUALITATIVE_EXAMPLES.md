# C-BARR Verbatim Qualitative Case Studies (Auto-Mined from Frozen Caches)

Curated verbatim examples for Section 5/6 figures and Appendix qualitative analysis.

## Case A: Overthinking Constraint Drift & C-BARR Recovery (`Qwen/Qwen3.5-4B`, `ifeval_3041`, slice=`ifeval_planning`)
- **Prompt**: ```` Write a creative and persuasive startup pitch for a business that helps people with chronic illnesses find and manage in-home care. Your pitch should be less than 7 sentences and contain exactly 1 bullet point in markdown. ````
- **Gate-1 Extracted Clause**: `phi_plan=1, phi_prec=0` | ```` Your pitch should be less than 7 sentences and contain exactly 1 bullet point in markdown. ````
- **B0 (`z=1`, `T=163`)**: ```` Imagine a world where chronic illness no longer means isolation, but rather a life filled with dignity and support right in the comfort of your own home. Our platform bridges the critical gap between patients and trusted caregivers by using AI-driven matching to connect you with professionals who truly understand yo... ````
- **B1b (`z=0`, `T=2048`, `loop=True`, `exhausted=True`)**:
  - `<think>` excerpt: ```` Thinking Process: 1. **Analyze the Request:** * **Topic:** Startup pitch for a business helping people with chronic illnesses find and manage in-home care. * **Tone:** Creative and persuasive. * **Constraint 1:** Less than 7 sentences (i.e., 6 sentences or fewer). * **Constraint 2:** Exactly 1 bullet point in markdo... ````
  - Final answer: `(empty)`
- **C-BARR M2 (`z=1`, `T=298`, `calls=1`)**: ```` Imagine a world where your health isn't a burden but a journey supported by a dedicated team right in your living room. Our platform connects you instantly with vetted caregivers who understand the unique rhythm of your chronic condition. We eliminate the guesswork and the anxiety by matching your specific needs wit... ````

## Case B: Budgeted Deliberation Gain Over Non-Thinking (`Qwen/Qwen3.5-4B`, `ifeval_2118`, slice=`ifeval_planning`)
- **Prompt**: ```` Write a rap for moms about the pros and cons of breast feeding versus formula. The rap song should have exactly 3 paragraphs each separated by *** and exactly 3 bullet points in markdown format. ````
- **Gate-2 Fast Probe**: `H_first=1.78, H_mean=1.68, H_max=3.72, I_inertia=0`
- **B0 (`z=0`, `T=409`)**: ```` Yo, listen up, it's time to talk about the choice, Between the natural bond and the bottle of choice. Some say the milk is pure, a gift from nature's hand, While others trust the science, the formula brand. It's a heavy topic, but we gotta be real, There are benefits on both sides that we need to... ````
- **C-BARR M2 (`z=1`, `T=456` vs `B1b T=2048`)**:
  - Capped `<think>`: ```` Thinking Process: 1. **Analyze the Request:** * **Topic:** Pros and cons of breastfeeding vs. formula (for moms). * **Format:** Rap song. * **Structure:** Exactly 3 paragraphs. * **Separator:** Each paragraph must be separated by `***`. * **Content Constraint:** Exactly 3 bullet points in markdow... ````
  - Final answer: ```` Yo, listen up, it's time to talk about the choice, Between the natural bond and the bottle of choice. Some say the milk is magic, packed with love and care, While others find the formula makes the journey easier to bear. It's a heavy decision, weighing every single side, There's no right or wrong... ````

## Case C: Section 6 Unstripped Echo Corruption (`B2b=0`) vs. Turn-Delimited Bridge (`B2a=1`) (`Qwen/Qwen3.5-4B`, `ifeval_1000`)
- **Prompt**: ```` Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*. ````
- **Control B2b (Unstripped Assistant Prefill, `z=0`)**: ```` [Reminder: strictly follow: Structure: Write a 300+ word summary of the wikipedia page "https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli". Do not use | Format: Do not use any commas and highlight at least 3 sections that has] *Raymond III Count of Tripoli* was a significant figure in the history of the Cr... ````
- **Control B2a (Turn-Delimited Post-Think Anchor, `z=1`)**: ```` *Raymond III Count of Tripoli* was a significant figure in the history of the Crusader states during the twelfth century. He held the title of Count of Tripoli from 1136 until his death in 1152. His father was Raymond II who established the county as a major stronghold in the Levant. This nobleman played a crucial r... ````

## Case D: Dense Softmax `<think>` KV Interference (`B2a=0`) vs. Zero-Token `<think>` KV Eviction (`z=1`) (`HuggingFaceTB/SmolLM3-3B`, `ifeval_3691`)
- **Prompt**: ```` Blog post from the perspective of a 16 year old girl who is being followed by a stranger. Your response should contain less than 10 sentences and no commas. ````
- **Text Re-Anchor Only (`B2a`, `z=0` due to `<think>` KV attention)**: ```` I'm being followed by a stranger I can't shake. He's always a step behind me. I try to ignore it but my heart races. I've been walking home from school and he's been lurking. I've tried to look away but he keeps following. I feel scared but I don't want to show it. I've been walking faster now. I... ````
- **Zero-Token `<think>` KV Cache Eviction (`evict_kv=True`, `z=1`)**: ```` I'm really scared right now. This guy keeps following me everywhere. I don't know what to do. I've tried to ignore him but he's getting closer. I need to tell someone but I don't want to make a scene. Maybe I should go to the police station. I hope they can help me. I just want to be safe. ````

## Case E: Out-of-Distribution `IFBench_test` Unseen Constraint Recovery (`Qwen/Qwen3.5-4B`, `ifbench_7_2850290c37`)
- **Instruction IDs**: `['count:conjunctions']`
- **Prompt**: ```` Use at least 3 different coordinating conjunctions in the response. Write in the prose, style, and tone of Finnigan's Wake. Consider the situation I am in; think about my father. Can I leave him, when he is blind and helpless, to deal with weakness and poverty, when I have the ability to make his old age comfortable and happy? ````
- **B1b (`z=0`, `T=2048`)**: `(empty)`
- **Frozen C-BARR M2 (`z=1`, `T=226`)**: ```` The wind howls through the hollows of the old house, a voice that speaks in riddles and whispers of a father whose sight is dimmed like a candle in a gale, yet whose spirit burns with a fire that cannot be extinguished by the cold of poverty or the gnawing teeth of weakness. Can I, who hold the k... ````

