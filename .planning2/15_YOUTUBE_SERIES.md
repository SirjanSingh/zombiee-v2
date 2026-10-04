# 15 — YouTube series plan: "AI learns to survive a zombie city"

**Created:** 2026-10-04 (manager session). **Inputs from Sirjan:** existing channel under 1K subs; goal = growth
first, monetisation later; AI voice (no face, no own voice); 1-2 videos/week if sustainable.

**How this was made:** installed and used two Claude skills, reviewed before install:
`claude-youtube` (AgriciDaniel, MIT, 417 stars) and the `yt-*` set (Jakeschincariol/youtube-agent-skill,
MIT, 477 stars). Ran `yt-viral`'s outlier method on real data pulled with yt-dlp (102 public videos from
5 genre channels) and scored hooks with `yt-script/hookscore.py`. Raw data:
scratchpad `ytdata/` (not committed).

---

## 1. What the genre data says

| channel | what it is | median length | median views (last ~25 videos) |
|---|---|---|---|
| AI Warehouse | RL agents, mascot "Albert" | 10.7 min (IQR 10.2-12.0) | 4.9M |
| b2studios | RL agents learn games/sports | 12.6 min | 427K |
| Gonkee | coding/physics/AI builds | 12.4 min | 401K |
| Code Bullet | AI learns games, comedic | 20.2 min | 2.4M |
| Emergent Garden | LLM agents in Minecraft | 20.6 min | 168K |

- **Length is not the lever.** Within each channel, views as a multiple of the channel median:
  5-10 min 1.12x, 10-20 min 0.99x, 20+ min 1.03x. The topic decides; length just has to fit the story.
- **Biggest outliers (views vs own channel median):** "AI Learns Insane Monopoly Strategies" 28x;
  "AI Learns To Swing Like Spiderman" 17x; **"4 AIs Survive 10 Days in Minecraft" 11x (LLM agents +
  survival + time limit = our premise)**; "AI Invents New Bowling Techniques" 9x; "AI Plays Minecraft
  Forever (and dies)" 3.9x.
- **Shared pattern:** a game everyone already knows + the AI does something a human would not
  ("insane strategies", "invents", "breaks the game"). The title promises emergent behaviour, not a method.

Benchmarks from the skill references (cited there; not re-verified by me):
- 55% of viewers leave in the first 60 s; 20% in the first 10 s. Retention at 30 s of 70%+ is solid.
- Entertainment optimal 8-12 min; 8 min unlocks mid-roll ads (~50% more revenue). Do not pad.
- Pattern interrupt every ~30 s for edited video. Open loops ("suspension bridge") = biggest completion lift.
- CTR: 4-6% average, 7-10% good. Channels under 500 subs get extra algorithmic testing.
- **AI narration is reported at ~70% lower retention** vs human-fronted, with an exception for explicitly
  AI-themed channels. This is the series' biggest risk (see section 4).

## 2. Format decisions

- **Long-form: 10-13 min**, one story per episode, payoff at the end. Matches AI Warehouse/b2studios and
  clears the 8-min mid-roll line without padding.
- **Shorts: 20-40 s**, cut from each episode's best moment (yt-shorts skill), 2-3 per long-form.
- **Cadence: 1 long-form every 2 weeks + 2-3 Shorts/week.** The entertainment template asks for 3/week
  and vidIQ's 5M-channel study links 12+/month to 8x view growth, but that is not sustainable solo on top
  of the actual training. Consistency beats volume (same study). Raise cadence only if a video breaks out.
- **A mascot.** Every successful channel here has a character (Albert, Code Bullet's persona). Give A0 a
  name and a look, keep it across episodes, and put its face in every thumbnail (the entertainment template
  says the face is the message; our face is the agent's).

## 3. Hooks (scored with hookscore.py; scorer calibrated on 74 short-form hooks, a guide not a promise)

Best after three rounds:
1. **"You're watching an AI starve to death on turn 21. It had water, food and a safehouse. So why did
   it walk outside?"** 55/100 WORKABLE (The Question). Opens on a visible failure, asks something only
   the video answers. Weakest: stakes.
2. "You can teach an AI to survive a zombie city. My first four tries all failed at zero percent, and the
   problem was never the AI." 44 WEAK (Impossible Claim), but true and leads into the "game was
   unwinnable" twist.

Lessons from the scoring: hooks that state the result ("it survives 72%") lose the curiosity score.
Hold the payoff back; show the failure first. Every number used is real (research_log/LOG.md).

## 4. AI-voice risk and how to handle it

- YouTube's inauthentic-content policy (renamed July 2025, clarified July 2026) does NOT ban AI voices;
  it demonetises mass-produced/template content and synthetic narration over stock footage with no
  original input. Ours is original research footage, so it should qualify, provided we tick the
  "altered or synthetic content" box (required for synthetic voices; disclosure does not reduce reach
  per the reporting). Sources: creatorblade.com, air.io, topuseai.com (see chat log 2026-10-04).
- Retention risk is real. Mitigations: the narrator is a character with a personality (written script,
  jokes, opinions), never generic; heavy on-screen action (replays of the actual game), interrupts every
  ~30 s; consider recording your own voice later for one episode and A/B the retention.

## 5. Episode arc (each beat is a real result already in research_log)

| ep | working title (genre pattern) | the real story | status |
|---|---|---|---|
| 1 | "I Taught an AI to Survive a Zombie City" | starves on turn 21, worse than doing nothing; 4 runs at 0%; the twist: the game itself was unwinnable even for a perfect bot; the traitor that "died of old age" | footage needed: replays (W5) |
| 2 | "My AI Beat the Bot That Taught It" | copying the planner -> starves at turn 25; DAgger -> 72% survival, teacher 0% | done, needs replays |
| 3 | "5 AIs, 1 Helicopter, 2 Are Infected" | radio + extraction, traitors, voting | W4 not built |
| 4 | "3 AI Training Methods Fight It Out" | GRPO vs GiGPO vs GAGPO, closed-loop | run 8 not built |
| 5 | "The AI Learned to Lie" (only if it happens) | self-play, model plays the infected too | future |

Shorts from what exists today: "AI starves in a room full of food", "the traitor died before it could
bite anyone", "the student beat its teacher".

## 6. Before the first upload (open items)

- Replay recorder (W5) and a game renderer: there is no footage yet; this is the blocking item.
- Write `~/.claude/youtube/voice.md` (the yt-script skill needs a voice profile; for an AI narrator it
  defines the character).
- Thumbnail + title as one pairing (`yt-package`), description/tags (`yt-seo`), chapters (`yt-chapters`).
- Production pipeline: `~/.claude/playbooks/video-production.md` (Hyperframes canvas engine, Kokoro TTS,
  numpy SFX, -14 LUFS master). Reference build `D:\projs\extra\nilesh`.
