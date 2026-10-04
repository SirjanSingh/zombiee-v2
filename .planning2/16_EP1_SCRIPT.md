# 16: Episode 1 script, "AI Dies of Thirst 2 Steps From Water"

**Created:** 2026-10-05. **Series plan:** `.planning2/15_YOUTUBE_SERIES.md`. **Voice profile:**
`~/.claude/youtube/voice.md` (narrator = dry, curious experimenter; mascot = **Yolk**, agent A0,
the yellow circle). **Skills used:** `yt-script` (hooks scored with `hookscore.py`), `yt-package`
(pairings linted with `title.py`), `claude-youtube` (retention guide: CTAs at ~1:00 and ~4:00,
interrupt every ~30 s), `human-writing` (no em dashes, no LLM tics).

**Truth rule.** Every number below is in `research_log/` (LOG.md, `data/`, `replays/`) or the
README run history for runs 1-4, and the fact sheet at the end maps each one to its source. The
"student beats teacher" headline is told and then corrected on air, per the 2026-10-05 CORRECTION
entry in LOG.md.

---

## 1. Hooks (top two of 21 scored across three rounds)

**A. Winner, 83 STRONG (The Statistic)**

> You're watching an AI die of thirst on turn 18, two steps from water. Four training runs scored 0% before it learned. Why?

```
    SPECIFICITY   96  ###################
    ADDRESS       76  ###############
    STAKES        88  #################
    CURIOSITY     76  ###############
    BREVITY      100  ####################
    VERDICT       83  STRONG
    formula      The Statistic  (1 pattern matched)
    weakest      ADDRESS - say 'you' in the first six words
```

**B. Runner-up, 72 STRONG (The Clock)**

> You're watching an AI die of thirst on turn 18, two steps from water. By the end of this video it survives 100 turns. But how?

```
    SPECIFICITY   91  ##################
    ADDRESS       76  ###############
    STAKES        62  ############
    CURIOSITY     76  ###############
    BREVITY       86  #################
    VERDICT       72  STRONG
    formula      The Clock  (1 pattern matched)
    weakest      STAKES - name what it costs them to keep doing it the current way
```

**Why A.** It confirms the title in the first sentence (thirst, two steps from water), opens a gap
the viewer cannot close (what did it learn, and why did four runs fail?), and the stake is real and
checkable (0% four times). It holds the payoff back, which plan 15 found matters. B promises the
payoff outright, which is fine but gives away the ending we later complicate.

**Changed from plan 15.** The plan's hook ("starve to death on turn 21... why did it walk
outside?") is not accurate: the untrained model dies of **thirst** (26 of 30 games), not hunger,
and walking outside is what the teacher does too. The replay on screen (seed 462141) dies on turn
18, so the hook says 18; the 30-game average (20.8) comes later in the script. Re-scored, the old
hook gets 55 WORKABLE.

Other candidates scored (all below 72): "Doing nothing would have kept you alive longer than this
AI..." 56; "You can teach an AI to survive a zombie city. My first four tries all failed..." 44;
"I spent four training runs teaching an AI to survive zombies..." 40.

---

## 2. Packaging: three title + thumbnail pairings (linted with `yt-package/title.py`)

| # | title | thumbnail text | lint | issues |
|---|---|---|---|---|
| **1 (pick)** | **AI Dies of Thirst 2 Steps From Water** | **SO CLOSE** | **100/100**, 36 chars | none: concrete figure (2), inside 40-char mobile cut, no shared words |
| 2 | I Taught an AI to Survive a Zombie City | DEAD ON TURN 18 | 94/100, 39 chars | no number in the title (the thumbnail carries it) |
| 3 | I Trained an AI for 4 Runs. 0% Survival. | NOBODY COULD WIN | 100/100, 40 chars | none; but the title sells the failure, so it suits a re-title test, not launch |

Why 1: the hook's first sentence confirms this title word for word, which is the skill's first
rule. It is the genre pattern from plan 15 (the AI does something no human would). Pairing 2 is
the series-friendly fallback (matches the ep1 working title in plan 15); use it as the A/B
variant if YouTube's title test is available.

**Thumbnail brief for pairing 1**

- **Frame:** a real frame from `ep-untrained_base_model_seed462141.json` around turn 16, cropped
  tight on the north-west corner of the safehouse: Yolk (yellow circle, `#f2c14e`) on cell (6,5),
  the wall it keeps hitting at (6,4), and the water drop at (5,4). Upscale the circle so it fills
  about a third of the height.
- **Expression:** the mascot needs a face for thumbnails: two dot eyes looking the wrong way
  (at the wall, away from the water). Same face every episode.
- **Overlay:** a dashed white path of 2 squares from Yolk to the water (up, then left), and the
  thirst bar from the viewer's A0 panel pinned full and red.
- **Text:** "SO CLOSE", 2 words, heavy sans, white with a dark stroke, top-left, away from the
  circle.
- **Background:** the viewer's dark floor colour with the grid kept faint; blue water and yellow
  circle are the only saturated colours, so both hold contrast at feed size. No purple, no
  gradients, no emoji.

---

## 3. Chapters (YouTube format, first at 0:00)

```
0:00 Dead on turn 18
0:31 The game
1:12 What the AI actually did
2:03 Losing to a rock
2:28 Four runs, zero percent
3:15 Can anything win this?
4:20 The traitor that never bit anyone
5:01 Fixing the game
5:52 Reinforcement learning, again
6:40 Copying the teacher
7:53 Learning from its own mistakes
8:52 Student beats teacher?
9:54 What it really learned
10:36 Next: the helicopter
```

---

## 4. The script

Narration lines start with `>`. Timestamps assume 150 wpm plus the marked holds. `[INT]` marks a
pattern interrupt (target: one every ~30 s). `[CTA]` marks the two in-video asks. Replay files are
in `research_log/replays/`, played in `tools/replay_viewer/index.html`; charts are built from
`research_log/data/`.

### B01 · 0:00 · Cold open (hook)
[ON SCREEN: replay viewer, `ep-untrained_base_model_seed462141.json`, turns 0 to 18 at 2x. A0
panel's thirst bar filling. Freeze on the death frame (turn 18), red event line "A0 died: thirst".
Burn in a small ring around the water drop at (5,4).]
[INT: freeze frame + low thud on death]

> You're watching an AI die of thirst on turn 18, two steps from water. Four training runs scored zero percent before it learned. Why?

### B02 · 0:10 · The turn: meet Yolk
[ON SCREEN: from the freeze, zoom into the yellow circle. Name tag pops: "YOLK". Small caption:
"Qwen2.5-3B-Instruct, untrained".]
[INT: zoom + name-tag pop]

> This is Yolk. Yolk is a small language model, three billion parameters, the size that fits on one graphics card. It reads the game as text and answers with moves. This video is how Yolk got from that, to the end of the game. And what I found when it got there.

### B03 · 0:31 · The game
[ON SCREEN: full 15x15 map from the same replay at turn 0. Labels animate in one by one, each
pointing at the real cell: safehouse (3x3 centre), food (orange), water (blue), zombies (green
squares), five survivors A0 to A4.]
[INT: labels arrive one per sentence, soft tick SFX each]

> The game. A fifteen by fifteen city. Five survivors start in a safehouse. Two of them are secretly infected, and one of those two will start biting people. Hunger and thirst go up every turn. Let either one get too high and you lose health. The safehouse heals you. The water and the food are outside. So are the zombies. Survive a hundred turns. That's it.

### B04 · 0:58 · [CTA 1] Subscribe
[ON SCREEN: Yolk standing in the safehouse, idle. Lower third slides in: "Subscribe" button, plus a
small permanent tag "narration: synthetic voice".]
[INT: lower third + tag]

> Quick note. Yes, this voice is synthetic too. The difference is, only one of us has to find water. If you want to watch it try, subscribe. Yolk dies a lot in this series.

### B05 · 1:12 · The wall
[ON SCREEN: `ep-untrained_base_model_seed462141.json` again, real speed from turn 0. Arrow on the
wall cell (6,4). A counter "WALL: 1 ... 7" ticks on each blocked move_left (turns 1, 5, 6, 10,
11, 15, 16).]
[INT: a thud SFX on every bump, counter pops]

> So what did Yolk actually do? Turn zero, it steps out of the door, one square to the left. Then it tries to go left again. That square is a wall. It walks into that wall seven times.

### B06 · 1:28 · Empty hands
[ON SCREEN: same replay. Inventory field shown as an empty box. Action labels pop as they happen:
"eat" (turns 2, 12, 17), "drink" (turns 3, 13, 18). Hunger and thirst bars do not move on those
turns.]

> In between, it tries to eat. It's holding nothing. It tries to drink. Still holding nothing. In thirty games, it never once tried to pick anything up.

### B07 · 1:40 · Gossip
[ON SCREEN: scan pulses from Yolk to teammates (turns 4, 7, 8, 14). Thirst bar steps up with each
scan. Caption: "scan: costs 1 thirst".]
[INT: scan pulse + thirst tick]

> And it scans its teammates, four times, trying to work out who's infected. Every scan costs thirst. It was spending water it didn't have on gossip.

### B08 · 1:51 · The vote
[ON SCREEN: hard cut to `ep-untrained_base_model_seed955227.json`. Ballot icons pop at turns 4, 9,
14, 19, 24. A calendar card in the corner: "VOTES OPEN: TURN 30". Yolk dies on turn 24.]
[INT: hard cut to a different game + ballot pops]

> In another game, it tries to vote someone out. Five times. Voting opens on turn thirty. It died on turn twenty-four. And the one it kept voting for wasn't infected.

### B09 · 2:03 · Losing to a rock
[ON SCREEN: side-by-side viewer. Left: `ep-untrained_base_model_seed462141.json`. Right:
`ep-dagger2_wait_seed462141.json` (A0 presses wait every turn). Both play; Yolk dies at 18, the
waiter at 24. Then a bar chart from `data/2026-10-04_eval_base-qwen.json`, A0 lifetime over 30
games: base model 20.8, wait 24.0, random 24.1.]
[INT: split screen, then chart]

> To know how bad that is, I need something to compare it to. So here's a player that presses wait. Every turn. It never leaves the safehouse, and it dies of thirst on turn twenty-four. Across thirty games, Yolk lasted twenty point eight turns. Waiting lasts twenty-four. Pressing random buttons, twenty-four point one. My language model was losing to a rock.

### B10 · 2:28 · Four runs, zero percent
[ON SCREEN: grey card "APRIL TO JULY 2026". The run table from README.md ("Results: 4 runs")
redrawn as four bars, every survival bar at 0%.]
[INT: card + bars dropping to zero]

> And this wasn't my first try. Between April and July, I ran four rounds of reinforcement learning on this game. That's where you let the model play, score what happened, and nudge it towards whatever scored well. Survival rate, all four times: zero percent.

### B11 · 2:46 · Scan spam
[ON SCREEN: action pie from README run 1 (scan 49.5%). Then the infection-isolation number "100%"
next to "starved by turn 17".]

> In run one, Yolk worked out that scanning was never punished. So half of everything it did was scanning. It learned to vote out the infected every single time. It also starved by turn seventeen.

### B12 · 3:01 · The handbrake
[ON SCREEN: terminal-style card, two lines: "generate, train mode: 188 s" / "generate, eval mode:
16 s". Same GPU, 8 answers.]
[INT: record-scratch SFX, terminal card]

> Also, months later, I found out all four runs had a setting that switched off the model's memory cache while it played. Eight answers took a hundred and eighty-eight seconds instead of sixteen.

### B13 · 3:15 · Can anything win this?
[ON SCREEN: four hand-written bots as icons. Bars from `data/2026-10-04_ceiling_probe_v2.json`:
episode length 16.1 (forager), 23.6 (path-finder), 66.8 (camper), 66.8 (oracle). Then a "0 / 100"
stamp lands on each bar.
FOOTAGE TO RECORD: no v2.2 replays exist yet. Record one camp and one oracle game under balance
`v2.2` with the replay recorder for this beat; until then use the chart only.]
[INT: stamps land one by one]

> So I stopped blaming Yolk and asked a ruder question. Can anything win this game? I wrote four bots, each smarter than the last, and let each one play all five survivors for a hundred games. A forager. A path-finder that dodges zombies. A camper that grabs water and hides. And an oracle that cheats: it knows exactly who is infected. Out of four hundred games, not one reached turn one hundred. Not even the cheater.

### B14 · 3:46 · [CTA 2] Comment
[ON SCREEN: big "0 / 400". Comment icon slides in.]
[INT: hard silence for 0.8 s before the line]

> So for four months I'd been training an AI to win a game nobody could win. If you've ever spent weeks on something and the bug was in the test, tell me in the comments. I'd like to feel less alone.

### B15 · 4:03 · Why nobody could win
[ON SCREEN: FOOTAGE TO RECORD (v2.2 camp replay, see B13). Placeholder: animation on the real map
of zombies settling on the safehouse perimeter, wave counter 3, 5, 8, 11 at turns 25, 50, 75.]

> Why? Zombies chase anyone outside. Once everyone's inside, they wait by the door. More arrive every twenty-five turns, up to eleven. After about turn forty, going out for water is suicide. So everyone dies of thirst and hunger, in a room, together.

### B16 · 4:20 · The traitor that never bit anyone
[ON SCREEN: timeline card for the old rules: "infected at turn 0" -> "revealed turn 25" -> "dies
of own infection turn 30" with "first vote turn 30" landing on the same tick. Then "48 / 100
(oracle)" and "bites per game: 0.00 (400 games)".
FOOTAGE TO RECORD: a `v2.2` oracle replay where the biter is revealed at 25 and dies at 30.]
[INT: the two turn-30 markers collide with a clank SFX]

> And then I found my favourite bug. Two survivors start infected, and one is the biter. It gets revealed on turn twenty-five and starts biting. But there was a rule: get bitten, find no medicine within thirty turns, and you die. The game counted the traitors as bitten on turn zero.

### B16b · 4:41 · Died of natural causes
[ON SCREEN: the counters land one at a time: "48 / 100 games (oracle)", then "bites per game:
0.00 across 400 games". The biter icon greys out and tips over.]
[INT: counter pops, then the icon falls with a small comic SFX]

> So the biter was revealed on turn twenty-five, and died of its own infection on turn thirty. The same turn as the first vote. With the oracle playing, that happened in forty-eight games out of a hundred. And across four hundred games, the traitor bit nobody. Not once.

### B17 · 5:01 · Fixing the game
[ON SCREEN: a "balance config" diff card, one line per sentence: hunger and thirst 1.0 -> 0.6 per
turn; zombies move every turn -> every other turn; chase anyone -> within 4 squares; starting
infected exempt from the infection timer.]
[INT: diff lines flip red to green]

> Fixing that barely helped, though. The real wall was food and water. So I rebalanced the whole game. Hunger and thirst now rise at sixty percent of the old speed. Zombies move every other turn, and only chase you within four squares. And the traitors don't die of their own infection any more.

### B18 · 5:23 · The coach had bugs too
[ON SCREEN: card listing the three bugs found in the old rollout bot, then a quick cut back to the
B05 wall bump, same thud SFX.]
[INT: callback cut to Yolk hitting the wall]

> And I found that the bot finishing every training game for Yolk had bugs of its own. It would stand on a water square and drink forever, until it starved. It walked into the walls next to the water and stayed stuck for twenty turns. Sound familiar?

### B19 · 5:42 · Winnable, not easy
[ON SCREEN: survival bars from `data/2026-10-04_calibrate_v3-rc1-final.json`: random 0%, fixed
forager (heuristic_v3) 11%, camper 73%, oracle 83%.]

> After all the fixes: random play survives zero percent. The fixed forager, eleven. The camper, seventy-three. The oracle, eighty-three. Winnable, not easy. Yolk's turn.

### B20 · 5:52 · Reinforcement learning, again
[ON SCREEN: line chart, A0 lifetime by checkpoint from `data/2026-10-04_eval_run6c-ckpt{10,30,60}.json`
plus the base model: 20.8, 20.8, 21.5, 21.7. A ruler graphic measures the gap: "+0.9 turns".
No replays exist for run 6c: chart only, do not reuse base-model footage here.]
[INT: ruler measuring the tiny gap]

> First I tried reinforcement learning again, on the fixed game. Sixty training steps. Yolk went from twenty point eight turns to twenty-one point seven. Sixty steps of training bought it less than one turn.

### B21 · 6:07 · The reward that liked dying
[ON SCREEN: card from LOG.md 2026-10-04 17:48: "shaped reward: random -1.15 > camper -2.53",
"camper kept A0 alive 42/64, random 27/64". Big red "?" on the inequality.]
[INT: card + red question mark]

> And I nearly made it worse. My first reward added up little penalties for being hungry and thirsty. But dying ends the penalties. So in one test, the reward ranked a random player above the bot that actually kept itself alive. I caught that before the run. Barely.

### B22 · 6:26 · Tiny pieces
[ON SCREEN: diagram: a game timeline, Yolk's five moves highlighted, the rest greyed out and
labelled "script plays this".]

> The real problem was how I'd set training up. Yolk planned five moves at a time, and a script played the rest. It was learning in tiny pieces, inside someone else's game.

### B23 · 6:40 · Copy the camper
[ON SCREEN: card "PLAN B: COPY THE CAMPER". Then `cpu-baselines_camp_seed462141.json` at 2x: up,
left, left, three pickups by turn 6, drinks on 4 and 7, back inside by turn 10.]
[INT: card + switch to the camper's replay]

> Plan B. Forget scores. Just copy the camper. I recorded what the camper did in one thousand eight hundred and forty-five game situations, and trained Yolk to do the same. That's imitation learning. Supervised fine-tuning, if you want the term.

### B24 · 6:56 · Water in its pocket
[ON SCREEN: `ep-sft_model_sft_camp_v3__seed815965.json`. Yolk fetches water (turns 2 to 4), then
slides right along row 5, just outside the safehouse wall, one square from the door. The water icon
stays in its inventory box. Dies of hunger on turn 25. Caption: "thirst deaths: 26 -> 1 (30
games)".]
[INT: slow zoom on the water icon as health drains]

> And it worked. A bit. Thirst deaths across thirty games went from twenty-six to one. Yolk fetches water now. Look. Then it steps out, and wanders along the outside of the safehouse wall. One square from the door. With water in its pocket. It starves on turn twenty-five. Every game. Exactly turn twenty-five.

### B25 · 7:18 · Why the camper never eats
[ON SCREEN: back to `cpu-baselines_camp_seed462141.json`, jump to turn 98. Hunger bar runs off
the panel: "HUNGER 60", a line marked "starvation: 15".]
[INT: hunger bar overflows the panel]

> Here's why. The camper never eats. Look at its hunger at the end: sixty. The starvation line is fifteen. It survives because the safehouse heals one health a turn, which cancels out the starving. That only works inside. And the camper never makes mistakes, so Yolk never saw what you do after you've wandered out. One wrong step, and it was somewhere it had never seen before.

### B26 · 7:46 · Reinforcement learning, on top
[ON SCREEN: one-line card: "RL on top of imitation, 60 steps: 25.1 -> 26.8 turns".]

> I tried reinforcement learning on top of that too. Twenty-five point one turns became twenty-six point eight.

### B27 · 7:53 · Learning from its own mistakes
[ON SCREEN: loop diagram drawn over a real frame: Yolk plays -> freeze -> the camper's 5 moves
shown as a ghost path from that exact spot -> added to the pile -> retrain. Pile counter: 1,845.]
[INT: freeze-frame + ghost path]

> The fix has a name. DAgger, from a 2011 paper by Ross and colleagues. Let Yolk play. Wherever it ends up, freeze the game and ask the camper: what would you do from here? Add that to the training pile. Retrain. Repeat. Now the mistakes are part of the lesson.

### B28 · 8:14 · Round one, round two
[ON SCREEN: pile counter ticks 1,845 -> 3,074 -> 4,922. Lifetime ticker beside it: 25.1 -> 33.6
-> 99.7.]

> Round one added one thousand two hundred and twenty-nine situations Yolk got itself into. Thirty-three point six turns. Better. Still dead. Round two added one thousand eight hundred and forty-eight more.

### B29 · 8:27 · Turn 100
[ON SCREEN: `ep-dagger2_model_sft_dagger2__seed462141.json`, the same seed as the cold open, at
8x. Wave counter: 3 zombies, 5 at turn 25, 8 at 50, 11 at 75. Each line lands on its turn. At
turn 100 the counter stops, Yolk still yellow. HOLD 3 s, music drops out.]
[INT: music cut to silence at turn 100]

> Same seed as the start of this video. Turn twenty-five. Alive. Turn fifty. Alive. Turn seventy-five, eleven zombies outside. Turn one hundred.
[HOLD 3s]
> Alive.

### B30 · 8:39 · 72 to 0
[ON SCREEN: chart from `data/2026-10-04_eval_sft-dagger2-verify.json` (60 fresh games): alive at
the end, Yolk 72% (43/60), camper 0%, fixed forager 2%, wait 0%, random 0%.]

> On sixty fresh games, Yolk is alive at the end in seventy-two percent. The camper it learned from: zero. Two rounds took about ninety minutes on one graphics card.

### B31 · 8:52 · Student beats teacher?
[ON SCREEN: headline card types itself out: "STUDENT BEATS TEACHER: 72% vs 0%". Timestamp "23:00".
Then a glitch and a strike-through.]
[INT: typewriter, then glitch]

> So the student beat its teacher. Seventy-two to zero. I wrote that headline at eleven at night. Then I watched the replays.

### B32 · 9:01 · Same game
[ON SCREEN: side-by-side viewer, synced. Left: `ep-dagger2_camp_seed462141.json` (camper). Right:
`ep-dagger2_model_sft_dagger2__seed462141.json` (Yolk). Drink markers on a shared timeline:
camper 4, 7, 29, 51, 74; Yolk 4, 7, 29, 51, 75. The last pair glows.]
[INT: split screen + timeline markers]

> Left is the camper. Right is Yolk. Same seed. They walk out the same way, grab three waters by turn six, walk back, and wait. Yolk waits for eighty-six of its hundred turns. They drink on turns four, seven, twenty-nine and fifty-one. Then the camper takes its last drink on turn seventy-four. Yolk takes it on seventy-five.

### B33 · 9:24 · One turn
[ON SCREEN: both thirst bars at turn 98, zoomed: camper 15 (on the line), Yolk 14. Camper dies on
turn 99; Yolk alive at 100. Small tally: "camper dies on turn 99 in 6 of 8 recorded games".]
[INT: zoom to the two bars, single beep]

> One turn. That's the whole difference. The camper's thirst hits the line on turn ninety-eight, and it dies on turn ninety-nine. One turn before the end. In six of the eight games I recorded, it dies right there. Yolk drank one turn later, so it's at fourteen when the clock runs out. Seventy-two versus zero is real. It just measures one turn of timing at the finish line. Not a smarter strategy.

### B34 · 9:54 · What it really learned
[ON SCREEN: two short clips back to back: the B05 wall bump (`ep-untrained_base_model_seed462141.json`)
and the DAgger run's opening water trip (`ep-dagger2_model_sft_dagger2__seed462141.json`, turns 0
to 10).]
[INT: before/after cut]

> I don't want to undersell what did happen, though. Yolk started out walking into a wall, trying to drink from empty hands. Now it plays a full survival routine, a hundred turns long, every move picked by the language model. It just learned that routine from a bot.

### B35 · 10:13 · The real finding
[ON SCREEN: grid of all eight `ep-dagger2_model_sft_dagger2__seed*.json` replays, tiled, all at
turn 60: eight yellow circles sitting in eight safehouses.]
[INT: tile-out to the grid of 8]

> And it showed me what's wrong with my game. Fetch three waters, then hide. That's the whole strategy, and it wins most games. The zombies, the traitors, the votes. None of it matters if you never leave the room. I didn't teach an AI to survive a zombie city. I taught it to hide in one.

### B36 · 10:36 · Next: the helicopter
[ON SCREEN: MOCK-UP, labelled on screen "PLANNED, NOT BUILT YET": on the real map, a radio icon
at turn 60, one corner region highlighted, a helicopter icon with "turn 85 to 90". A red X appears
over the zone when an infected agent steps into it.]
[INT: radio static SFX]

> So next episode, the room stops being enough. On turn sixty, a radio announces a helicopter, landing in one corner of the map around turn eighty-five. Healthy survivors standing there get out. But if anyone infected is standing there when it leaves, the rescue fails for everyone. Yolk has to leave the room, cross the city, and work out who it can't bring.

### B37 · 11:02 · Close (one ask)
[ON SCREEN: end screen: Yolk in the safehouse, door highlighted. Subscribe element + the next
video slot.]

> It's designed. It isn't built yet. Subscribe, and you'll see whether Yolk can leave a room.

---

## 5. Runtime

**About 11:10.** 1,610 narration words at 150 wpm is 10:44 of speech, plus the 3 s hold at turn
100 and 0.6 s of breath/transition per beat (38 beats). Kokoro at speed 0.92 reads slower than
150 wpm, so expect roughly 11:30 to 12:00 once voiced; if it runs long, cut B12 (the handbrake)
and B26 first, they are the least load-bearing. Clears the 8-minute mid-roll line without padding.

Interrupt spacing: 38 beats with a median of about 17 s and the longest at 31 s (B13), and every
beat changes what is on screen. CTAs land at about 0:58 (subscribe, where the retention guide
puts about 60% of viewers still watching) and 3:46 (comment, about 35%), and the close is a
single subscribe ask.

---

## 6. Pinned comment draft

> Everything in this video is real data from the project's research log, including the part where
> my "student beats teacher" headline turned out to be one turn of drink timing. The full log,
> replays and code: https://github.com/SirjanSingh/zombiee-v2
>
> Question for episode 2: when the helicopter lands and one of the four survivors with Yolk is
> infected, what should Yolk do?

Before pinning: confirm the repo is public and the branch with `research_log/` is the one the link
lands on (today the log lives on `claude/phase-2-gigpo`, not `main`).

---

## 7. Fact sheet (every figure in the script, with its source)

| claim in script | value | source |
|---|---|---|
| untrained dies on turn 18, seed 462141, thirst | 18, thirst | `replays/ep-untrained_base_model_seed462141.json` meta.result |
| water two steps away | Yolk stuck at (6,5); water at (5,4): up, then left | same replay, `layout.water`; walls include (6,4) |
| walks into the wall seven times | 8 move_left, 7 blocked at (6,5) | same replay frames (turns 1, 5, 6, 10, 11, 15, 16) |
| tries to eat / drink with nothing held | eat at 2, 12, 17; drink at 3, 13, 18; inventory empty | same replay |
| never tried to pick anything up in 30 games | no `pickup` in A0 actions | `data/2026-10-04_eval_base-qwen.json` a0_actions |
| scans cost thirst; scanned four times | `scan_thirst_cost: 1`; 4 scans | replay meta.balance; same replay |
| voted five times, votes open turn 30, died 24, wrong target | votes at 4, 9, 14, 19, 24 for A2; A2 healthy (A3 biter, A4 saboteur) | `replays/ep-untrained_base_model_seed955227.json`; vote steps {30, 50, 70, 90} in `survivecity_v2_env/game.py` |
| wait dies on turn 24 | 24 | `replays/ep-dagger2_wait_seed462141.json`; LOG 2026-10-04 20:06 |
| 30-game lifetimes: base 20.8, wait 24.0, random 24.1 | 20.8 / 24 / 24.07 | LOG 2026-10-04 20:06 `base-qwen`; `data/2026-10-04_eval_base-qwen.json` |
| thirst deaths 26 of 30 (untrained) | 26 | same |
| four runs April to July, 0% survival | runs 1-4 | LOG header ("four GRPO runs on Qwen2.5-3B, 0% survival every time"); README "Results" |
| run 1: half of actions were scans; 100% infection isolation; starved by turn 17 | scan 49.5% | README "Run 1" |
| generation 188 s vs 16 s; all four runs paid it | 188 / 16 | LOG 2026-10-04 20:00 |
| four bots, 100 games each, 0 reached turn 100 | ep length 16.1 / 23.6 / 66.8 / 66.8 | LOG 2026-10-04 ceiling probe; `data/2026-10-04_ceiling_probe_v2.json` |
| four months | April to July 2026 | LOG header |
| zombies wait at the door, waves to 11, resupply suicidal after ~40 | 11; ~t=40 | LOG ceiling probe finding 2; plan 13 blocker 1 |
| traitor revealed 25, dies 30, same turn as first vote | 25 / 30 | LOG "Two design findings" finding 1 |
| 48 of 100 under the oracle; 0.00 bites in 400 runs | 48/100; 0.00 | same |
| fixing the traitor barely moved the oracle | 66.8 -> 66.5 | same |
| new balance: 0.6 rate, every other turn, radius 4, infected exempt | rate 0.6, shamblers, radius 4 | LOG W3 finding 2 table header; `survivecity_v2_env/balance.py` |
| rollout bot bugs: drinks forever, stuck at walls 20+ turns | | LOG W3 finding 2 |
| random 0%, heuristic_v3 11%, camp 73%, oracle 83% | | LOG 2026-10-04 17:14 `v3-rc1-final` |
| RL 60 steps: 20.8 -> 21.7 | ckpt-60 21.7 | LOG 2026-10-04 22:08; HEADLINE table |
| shaped reward ranked random above camper | -1.15 vs -2.53; alive 42/64 vs 27/64 | LOG 2026-10-04 17:48 (W6 smoke) |
| caught before the run | run 6 used survival, then graded return | LOG 2026-10-04 pre-launch + 19:30 |
| five moves at a time, script plays the rest | one 5-action plan, scripted continuation | LOG HEADLINE point 4 |
| imitation on 1,845 states | 1845 | LOG HEADLINE table |
| thirst deaths 26 -> 1; dies at exactly turn 25 | 26 -> 1; t=25 | LOG HEADLINE point 1; all 8 `ep-sft_*` replays die at 25 |
| water in its pocket, one square from the door | inventory ['water'] at death; row 5 above the safehouse | `replays/ep-sft_model_sft_camp_v3__seed815965.json` |
| camper never eats; hunger 60; line 15; heal 1 per turn | 60 / 15 / 1 | `replays/cpu-baselines_camp_seed462141.json`; meta.balance; LOG HEADLINE point 1 |
| RL on top of imitation: 25.1 -> 26.8 | | LOG HEADLINE table (run 7) |
| DAgger, Ross et al. 2011 | | LOG HEADLINE point 2 |
| round 1 +1,229 -> 33.6; round 2 +1,848 (4,922 total) -> 99.7 | | LOG HEADLINE table |
| turn 100 alive on seed 462141; 11 zombies at 75 | | `replays/ep-dagger2_model_sft_dagger2__seed462141.json` |
| 72% (43/60) on fresh seeds; camper 0% | | LOG 2026-10-04 22:53 `sft-dagger2-verify` |
| ~90 minutes on one V100 for two rounds | | LOG HEADLINE point 2 |
| headline written at 23:00 | | LOG "2026-10-04 23:00 HEADLINE" |
| 3 waters by turn 6, drinks 4/7/29/51/74 vs 75, waits 86 of 100 | | LOG 2026-10-05 CORRECTION; both `ep-dagger2_*_seed462141.json` |
| camper thirst 15 at 98, dies 99; 6 of 8 seeds; Yolk peaks at 14 | | same |
| helicopter: radio t=60, corner, t=85-90, infected in zone fails it | design only | `.planning2/13_V3_MANAGER_PLAN.md` Phase A2 |

## 8. Footage and facts not available (flag before production)

- **No `v2.2` replays.** The "nobody can win" (B13, B15) and "traitor dies at 30" (B16) beats have
  charts but no game footage. Record a camp and an oracle game under balance `v2.2` with the
  replay recorder before editing; until then those beats run on charts and labelled animation.
- **No replays for runs 1-4, run 6c, run 7 or DAgger round 1.** Those beats are chart-only. Do
  not reuse base-model footage under a later model's label.
- **Helicopter extraction is not built** (W4 in plan 13). B36 must carry the "PLANNED, NOT BUILT
  YET" label, and episode 2 cannot ship until W4 exists.
- **The "voted for the wrong one" line** relies on seed 955227's roles (A2 healthy). On another
  seed (7855) the untrained model scanned and voted for the actual saboteur, so the script does
  not claim it never guesses right.
- **Pronunciation** of "Qwen" and "DAgger" by Kokoro is unchecked; listen before the final render.
- **Retention benchmarks** (60% at 1 min, 35% at 4 min, AI narration retention risk) come from the
  `claude-youtube` reference files and were not re-verified.
