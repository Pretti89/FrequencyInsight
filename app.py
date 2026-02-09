import gradio as gr
import librosa, numpy as np
import matplotlib.pyplot as plt
import tempfile, os, shutil, traceback, re, hashlib, json
import yt_dlp
import whisper
import torch
import soundfile as sf
from functools import lru_cache
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ======================================================
# SETTINGS (BANDS / COLORS / HUMAN EXPLANATIONS)
# ======================================================

BANDS = {
    "Red": (20, 250),
    "Orange": (250, 500),
    "Yellow": (500, 2000),
    "Green": (2000, 4000),
    "Blue": (4000, 6000),
    "Indigo": (6000, 10000),
    "Violet": (10000, 20000)
}

COLORS = {
    "Red": "#FF4C4C",
    "Orange": "#FF9F1C",
    "Yellow": "#FFD93D",
    "Green": "#6BCF63",
    "Blue": "#4D96FF",
    "Indigo": "#6A4C93",
    "Violet": "#B983FF"
}

# Keep your long human descriptions (NO REDUCTION)
FREQ_HUMAN = {
    "Red": {
        "range": "20–250 Hz",
        "effect": (
            "Physical drive, grounding, body activation, visceral impact, weight, density, pressure, pulse, "
            "instinct, rootedness, solidity, momentum, intensity, muscular engagement, somatic focus, "
            "stomp-like power, chest resonance, bodily urgency, primal alertness, forcefulness, "
            "heavy groove, internal rumble, heart-thump sensation"
        ),
        "benefit": (
            "Energy, stamina, confidence, decisiveness, resilience, physical motivation, embodiment, "
            "action-readiness, willpower, courage, steadiness, grit, endurance, stable presence, "
            "movement support, athletic push, productivity fuel, grounded focus, assertiveness, "
            "drive to initiate, stronger body awareness, commitment, persistence"
        ),
        "risk": (
            "Agitation, tension, irritability, impulsivity, fatigue, restlessness, stress build-up, "
            "aggression, tight jaw/shoulders, nervous activation, unease, overheating, racing body, "
            "difficulty relaxing, stubborn mood, edgy energy, pressure headache, overdrive, "
            "physical drain, sleep disruption, anxiety amplification, crankiness, sensory overload"
        )
    },
    "Orange": {
        "range": "250–500 Hz",
        "effect": (
            "Warmth, groove, rhythmic motion, emotional flow, vibrancy, social energy, body sway, "
            "bounce, swing, musical 'body', fullness, chesty warmth, movement impulse, danceability, "
            "playful momentum, organic texture, lively mid-bass, emotional color, kinetic feel, "
            "forward motion, friendly presence, smooth drive"
        ),
        "benefit": (
            "Motivation, enjoyment, creativity, sociability, openness, emotional expression, "
            "positive engagement, playful mood, connection, comfort, flow-state support, "
            "confidence in movement, mood lift, friendliness, cooperation, expressive energy, "
            "creative spark, spontaneity, light vitality, enthusiasm, warmth, charm, ease"
        ),
        "risk": (
            "Restlessness, impatience, distraction, nervous excitement, fidgety energy, "
            "inconsistent focus, mood swings, overstimulation, impulsive decisions, "
            "irritability, agitation creep, tension buildup, buzzing body, irritative warmth, "
            "fatigue from constant groove, difficulty settling, mental drift, emotional volatility, "
            "uneasy excitement, over-caffeinated feeling, sleep delay, inner agitation"
        )
    },
    "Yellow": {
        "range": "500–2000 Hz",
        "effect": (
            "Attention, cognition, alertness, language focus, analytical activation, mental speed, "
            "thought density, vocal intelligibility, conceptual clarity, problem-solving mode, "
            "planning mindset, detail-tracking, precision, mental sharpness, cognitive push, "
            "information processing, task orientation, focus lock, mental 'light on', "
            "logic-forward tone, decision pressure, brain-on feeling, concentration"
        ),
        "benefit": (
            "Focus, clarity, learning, productivity, comprehension, studying support, "
            "reading stamina, logical thinking, memorization help, mental organization, "
            "decision-making, disciplined work, coding/debugging support, planning, "
            "structured thought, motivation to finish tasks, attention stability, "
            "efficiency, clarity of speech, problem solving, sharper recall, mental stamina"
        ),
        "risk": (
            "Overthinking, anxiety, cognitive overload, mental fatigue, tension, "
            "racing thoughts, sleep disruption, irritability, hypervigilance, "
            "stress loop, inability to switch off, perfectionism spike, head pressure, "
            "restless mind, reduced calm, compulsive analysis, burnout risk, "
            "focus rigidity, emotional dryness, agitation from constant stimulation, "
            "decision exhaustion, brain fog after long exposure"
        )
    },
    "Green": {
        "range": "2000–4000 Hz",
        "effect": (
            "Balance, perceptual clarity, harmonic coherence, stability, ease of listening, "
            "definition, presence clarity, crispness without harshness, organized sound field, "
            "mid-detail structure, speech clarity band, alignment, tonal accuracy, "
            "clean edges, musical steadiness, coherent layering, stable texture, "
            "clarity anchor, grounded detail, smooth articulation, centered perception"
        ),
        "benefit": (
            "Comfort, harmony, stable perception, long listening tolerance, "
            "balanced mood, reduced fatigue, easier comprehension, stable attention, "
            "calm clarity, reliable background listening, emotional steadiness, "
            "smooth detail, improved intelligibility, gentle focus, "
            "pleasant neutrality, steadiness for work, less chaotic feeling, "
            "clean listening experience, consistent tone, fewer spikes, equilibrium"
        ),
        "risk": (
            "Emotional flatness if dominant, muted excitement, reduced intensity, "
            "low emotional color, boredom, under-stimulation, detached feel, "
            "too neutral, lack of motivation, blandness, decreased arousal, "
            "sleepiness (for some), less drive, reduced urgency, "
            "monotony, low engagement, “background blur”, limited emotional push, "
            "uninspiring tone, passive mood, mild dullness, low spark"
        )
    },
    "Blue": {
        "range": "4000–6000 Hz",
        "effect": (
            "Calm detail, subtle awareness, emotional nuance, reflective tone, gentle clarity, "
            "soft edges, fine articulation, airy presence, light shimmer, calm attention, "
            "quiet precision, soothing detail, delicate texture, emotional softness, "
            "cool calm, stable quiet, mindful listening band, soft focus, "
            "relaxed perception, subtle dynamics, serene tone, gentle space"
        ),
        "benefit": (
            "Relaxation, emotional regulation, calm focus, decompression, "
            "mindfulness support, calmer breathing, soothing background, "
            "gentle attention, emotional smoothing, stress relief, "
            "better downshift after work, reflective thinking, calm clarity, "
            "less agitation, peaceful ambience, soft concentration, "
            "relaxed mood, gentle introspection, reduced tension, sleep wind-down, "
            "cool-headedness, emotional balance, comfort"
        ),
        "risk": (
            "Low alertness, sleepiness, reduced drive, passivity, "
            "slower reaction, lowered energy, reduced motivation, "
            "too calm for task urgency, zoning out, daydream drift, "
            "loss of intensity, decreased engagement, dull focus, "
            "mental wandering, procrastination tendency, "
            "over-relaxation, fatigue feeling, “soft crash”, attention drop, "
            "difficulty staying sharp, under-stimulation, sluggishness"
        )
    },
    "Indigo": {
        "range": "6000–10000 Hz",
        "effect": (
            "Texture, imagination, abstraction, symbolic thinking, atmosphere, "
            "sparkle detail, complex timbre, layered air-texture, vivid nuance, "
            "creative stimulation, sonic complexity, dreamlike color, "
            "mental imagery, heightened texture perception, impressionistic feel, "
            "mystery tone, surreal edge, artistic detail, shimmering grit, "
            "fine grain, mental movement, “idea swirl”, vivid sound particles"
        ),
        "benefit": (
            "Creativity, inspiration, artistic exploration, idea generation, "
            "visual imagination, novel thinking, abstraction, experimentation, "
            "sound design appreciation, deeper texture awareness, "
            "creative mood, storytelling fuel, artistic flow, "
            "inventiveness, expressive writing support, imaginative play, "
            "fresh perspective, innovation energy, brainstorming support, "
            "curiosity, conceptual exploration, emotional artistry, nuance"
        ),
        "risk": (
            "Mental fatigue, sensory overload, irritation (at volume), "
            "tension from detail density, restless mind, headache risk, "
            "overstimulation, distraction, anxious buzzing, "
            "sharpness fatigue, focus fragmentation, nervous irritation, "
            "sleep disruption, ear strain, reduced calm, "
            "detachment, dissociation-like drift, “too much detail”, "
            "wired tiredness, agitation creep, overstressed attention, "
            "irritability, burnout with looping"
        )
    },
    "Violet": {
        "range": "10000–20000 Hz",
        "effect": (
            "Air, spaciousness, shimmer, brilliance, heightened perception, "
            "bright edge, crisp air band, glittering top, clarity halo, "
            "sparkling presence, open space feel, airy lift, "
            "high-frequency sparkle, breathiness, sheen, shining top-end, "
            "space expansion, delicate sparkle, glassy detail, "
            "bright openness, high shimmer, airy texture, light aura"
        ),
        "benefit": (
            "Lightness, openness, refined clarity, spacious feel, "
            "freshness, perceived detail, airy lift, "
            "clean brilliance, sense of space, “open window” feeling, "
            "clarity enhancement, refined listening, delicate beauty, "
            "atmospheric openness, crisp sparkle, polished sound, "
            "breathiness, shimmer comfort (when smooth), clarity glow, "
            "liveliness, airy calm (when balanced), expansive mood, uplift"
        ),
        "risk": (
            "Irritation, listener fatigue, harshness, ear strain, "
            "sharpness, sibilance fatigue, headache trigger, "
            "over-bright discomfort, stress response, sensory overload, "
            "tension spikes, anxiety amplification (for some), "
            "sleep disruption, agitation, “glassy” discomfort, "
            "reduced warmth, thinness feeling, brittle edge, "
            "tiring to loop, overstimulated nervous system, irritation creep, "
            "fatigue at high volume, discomfort sensitivity"
        )
    }

}


NEGATIVE_EFFECTS = {
    "Red": "tension, overdrive, pressure build-up, bodily restlessness, difficulty calming down, edgy activation, jaw/shoulder tightness",
    "Orange": "inner agitation, impatience, unsettled mood, fidgety energy, difficulty settling, buzzy stimulation, scattered focus",
    "Yellow": "mental pressure, overthinking tendency, cognitive overload, racing thoughts, attention fatigue, decision exhaustion, sleep delay",
    "Green": "emotional flatness, muted excitement, low emotional contrast, reduced spark, background-like feel, under-stimulation, dullness",
    "Blue": "sluggishness, slowed momentum, low urgency, passivity, reduced drive, comfort-over-action bias, sleepy drift",
    "Indigo": "sensory density, detail overload, listening strain, mental heaviness, headache-prone focus, restless perception, fatigue build-up",
    "Violet": "spaced-out feeling, dreamy drift, detachment, foggy presence, low grounding, sleepiness tendency, airy overload"
}

# ======================================================
# LEXICONS (keep/add later; unchanged structure)
# ======================================================

DRUG_WORDS = set([
    "drug","drugs","weed","maryjane","marijuana","cannabis","hash","hashish","joint","blunt",
    "coke","cocaine","crack","meth","speed","adderall","xanax","perc","percocet","oxy","oxycodone",
    "heroin","dope","needle","snort","stash","high","stoned","pill","pills","molly","ecstasy","mdma",
    "lean","codeine","syrup","ketamine","k","lsd","acid","shrooms","mushrooms","trip","tripping",
    "fent","fentanyl","narcan","overdose", "putaria", "porra","caralho","puta","putaria","fuder","foda","merda","cuzão","cuzao","buceta","piranha","vadia", "putaria", "safada", "bandida",
    "macho", "maxo", "cachorra", "cachorro", "oitao", "otario", "bereta", "os hommi", "comeca a botar", "sentando", "de ladinho", "safada",
    "safado", "raba", "rabao", "chupa", "chupada", "chupando", "chupa chupa",
    "cu", "sentar", "quicar", "rebolar",
    "descer", "foder", "gemer", "botar", "dar", "safada", "putaria",
    "tesão", "gostosa", "bundão", "rabão", "novinha", "piranha",
    "cachorra", "vagabunda", "filho da puta", "otário", "trouxa",
    "corno", "pau no cu", "arrombado", "babaca", "patrão", "chefe",
    "bandida", "maloka", "cria", "mandrake", "vida loka", "postura",
    "presença", "respeito", "baile", "paredão", "pancadão", "grave",
    "grave batendo", "tremendo", "rajada", "estourado", "fluxo",
    "passinho", "B.O.", "glock", "blindado", "envolvido", "fita",
    "proceder", "responsa", "quadrilha", "disposição", "no corre",
    "chavosa", "chavoso", "marra", "marrenta", "pesadão", "brabo",
    "sinistro", "nervoso", "visão", "visão de cria", "no grau",
    "na maldade", "no pique", "no talento", "no fluxo", "no baile",
    "fogo", "malemolência", "sedução", "tentação", "vem", "vai",
    "joga", "bota", "sobe", "desce", "encosta", "toma", "vira",
    "chama", "agora", "hoje", "sempre", "nunca", "tudo", "nada",
    "muito", "tudo nosso", "favela", "quebrada", "sarrar", "sarração", "encaixar", "encaixe", "jogada", "mexer",
    "mexida", "mexe", "chacoalhar", "balançar", "rebolada", "sentada",
    "quicada", "travada", "empinar", "empinada", "deslizar", "escorregar",
    "colada", "coladinha", "pele", "suor", "calor", "pegada", "pegador",
    "pegadora", "malícia", "safadeza pura", "sem limite", "sem freio",
    "sem censura", "sem pudor", "ousada", "ousado", "atrevimento total",
    "liberdade", "libertina", "libertino", "luxúria", "prazer",
    "prazerosa", "prazeroso", "excitação", "ardente", "intenso",
    "intensa", "provoca", "provocação", "dominar", "dominante",
    "dominada", "controle", "sem controle", "perigo", "tentador",
    "tentadora", "instinto", "impulso", "clima", "climinha",
    "vibe pesada", "vibe quente", "pressão", "pressão total",
    "impacto", "choque", "batida seca", "grave pesado", "subgrave",
    "sub pesado", "batendo forte", "estrondo", "explodir", "explosão",
    "quente demais", "suando", "suadinha", "energia", "energia bruta",
    "presença forte", "domínio", "liderança", "moral", "conceito",
    "conceituada", "conceituado", "estilo próprio", "postura firme", "mandela", "mandelao",
    "mandelinha", "autenticado", "paga de grandao", "chupa chupa", "chupa-chupa", "chupachupa",
    "chup", "pipokinha", "pipoquinha", "espancamento", "sessao de espancamento", "espancar",
    "gay", "nazi", "facismo", "fascist", "facista", "fascista", "minoria", "minority", "demonio", "demon", "devil","demônio",
    "estupram", "estrupar", "estrupamento", "homofóbico"
])

VIOLENCE_WORDS = set([
    "kill","killing","murder","murdered","shot","shoot","shooting","gun","glock","pistol","rifle",
    "knife","stab","stabbing","blood","bleed","bleeding","war","fight","fighting","assault",
    "beat","beating","punch","smash","terror","bomb","explode","explosion","dead","death",
    "choke","strangle","rage","violent","violence""Buceta", "piroca", "pau", "rola", "pica", "xoxota", "xana", "cu",
    "arrombado", "foder", "sentar", "rebolar", "descer", "quicar", "botar",
    "dar", "socar", "gemer", "tapão", "safada", "filho da puta",
    "vagabunda", "piranha", "cachorra", "otário", "babaca", "trouxa",
    "corno", "pau no cu", "arrombado (insulto)", "maloka", "maloqueiro",
    "mandrake", "cria", "favela venceu", "menor", "patrão", "vida loka",
    "chefe", "bandida", "novinha", "B.O.", "blindado", "Glock", "rajada",
    "envolvido", "fita", "proceder", "quadrilha", "responsa", "disposição",
    "Safadeza", "putaria", "putão", "putona", "tarado", "tarada", "tesão",
    "tesuda", "gostosa", "gostoso", "rabão", "bundão", "bunduda", "peitão",
    "peituda", "maldade", "malicioso", "maliciosa", "ousadia",
    "provocante", "descarada", "descarado", "sem vergonha", "atrevida",
    "atrevimento", "fogo", "no grau", "no ponto", "na maldade",
    "no talento", "no pique", "no corre", "no fluxo", "no baile",
    "no passinho", "tremedeira", "pancadão", "paredão", "grave",
    "grave batendo", "tremendo", "estourado", "estourada", "chavosa",
    "chavoso", "chavão", "chavinha", "chave", "chavezinha", "marra",
    "marrenta", "marrento", "postura", "presença", "respeito", "visão",
    "visão de cria", "sangue bom", "sinistro", "pesadão", "pesadona",
    "brabo", "braba", "brabíssimo", "nervoso", "nervosinha", "avançada",
    "avançado", "envolvente", "envolvência", "malemolência", "sedução",
    "tentação", "proibidão", "pescoço duro", "bolado", "boladona",
    "boladão", "desenrolado", "desenrolada", "fechamento", "fechado",
    "fortão", "fortinha", "putaria", "porra","caralho","puta","putaria","fuder","foda","merda","cuzão","cuzao","buceta","piranha","vadia", "putaria", "safada", "bandida",
    "macho", "maxo", "cachorra", "cachorro", "oitao", "otario", "bereta", "os hommi", "comeca a botar", "sentando", "de ladinho", "safada",
    "safado", "raba", "rabao", "chupa", "chupada", "chupando", "chupa chupa",
    "cu", "sentar", "quicar", "rebolar",
    "descer", "foder", "gemer", "botar", "dar", "safada", "putaria",
    "tesão", "gostosa", "bundão", "rabão", "novinha", "piranha",
    "cachorra", "vagabunda", "filho da puta", "otário", "trouxa",
    "corno", "pau no cu", "arrombado", "babaca", "patrão", "chefe",
    "bandida", "maloka", "cria", "mandrake", "vida loka", "postura",
    "presença", "respeito", "baile", "paredão", "pancadão", "grave",
    "grave batendo", "tremendo", "rajada", "estourado", "fluxo",
    "passinho", "B.O.", "glock", "blindado", "envolvido", "fita",
    "proceder", "responsa", "quadrilha", "disposição", "no corre",
    "chavosa", "chavoso", "marra", "marrenta", "pesadão", "brabo",
    "sinistro", "nervoso", "visão", "visão de cria", "no grau",
    "na maldade", "no pique", "no talento", "no fluxo", "no baile",
    "fogo", "malemolência", "sedução", "tentação", "vem", "vai",
    "joga", "bota", "sobe", "desce", "encosta", "toma", "vira",
    "chama", "agora", "hoje", "sempre", "nunca", "tudo", "nada",
    "muito", "tudo nosso", "favela", "quebrada", "sarrar", "sarração", "encaixar", "encaixe", "jogada", "mexer",
    "mexida", "mexe", "chacoalhar", "balançar", "rebolada", "sentada",
    "quicada", "travada", "empinar", "empinada", "deslizar", "escorregar",
    "colada", "coladinha", "pele", "suor", "calor", "pegada", "pegador",
    "pegadora", "malícia", "safadeza pura", "sem limite", "sem freio",
    "sem censura", "sem pudor", "ousada", "ousado", "atrevimento total",
    "liberdade", "libertina", "libertino", "luxúria", "prazer",
    "prazerosa", "prazeroso", "excitação", "ardente", "intenso",
    "intensa", "provoca", "provocação", "dominar", "dominante",
    "dominada", "controle", "sem controle", "perigo", "tentador",
    "tentadora", "instinto", "impulso", "clima", "climinha",
    "vibe pesada", "vibe quente", "pressão", "pressão total",
    "impacto", "choque", "batida seca", "grave pesado", "subgrave",
    "sub pesado", "batendo forte", "estrondo", "explodir", "explosão",
    "quente demais", "suando", "suadinha", "energia", "energia bruta",
    "presença forte", "domínio", "liderança", "moral", "conceito",
    "conceituada", "conceituado", "estilo próprio", "postura firme", "mandela", "mandelao",
    "mandelinha", "autenticado", "paga de grandao", "chupa chupa", "chupa-chupa", "chupachupa",
    "chup", "pipokinha", "pipoquinha", "espancamento", "sessao de espancamento", "espancar",
    "gay", "nazi", "facismo", "fascist", "facista", "fascista", "minoria", "minority", "demonio", "demon", "devil","demônio",
    "estupram", "estrupar", "estrupamento", "homofóbico"
])

SEXUAL_WORDS = set([
    "sex","sexual","fuck","fucking","f*ck","pussy","dick","cock","ass","nude","naked","cum",
    "orgasm","porn","porno","horny","slut","whore","bitch","booty","thong","twerk","blowjob",
    "anal","ride","bed","freak","kinky" "Buceta", "piroca", "pau", "rola", "pica", "xoxota", "xana", "cu",
    "arrombado", "foder", "sentar", "rebolar", "descer", "quicar", "botar",
    "dar", "socar", "gemer", "tapão", "safada", "filho da puta",
    "vagabunda", "piranha", "cachorra", "otário", "babaca", "trouxa",
    "corno", "pau no cu", "arrombado (insulto)", "maloka", "maloqueiro",
    "mandrake", "cria", "favela venceu", "menor", "patrão", "vida loka",
    "chefe", "bandida", "novinha", "B.O.", "blindado", "Glock", "rajada",
    "envolvido", "fita", "proceder", "quadrilha", "responsa", "disposição",
    "Safadeza", "putaria", "putão", "putona", "tarado", "tarada", "tesão",
    "tesuda", "gostosa", "gostoso", "rabão", "bundão", "bunduda", "peitão",
    "peituda", "maldade", "malicioso", "maliciosa", "ousadia",
    "provocante", "descarada", "descarado", "sem vergonha", "atrevida",
    "atrevimento", "fogo", "no grau", "no ponto", "na maldade",
    "no talento", "no pique", "no corre", "no fluxo", "no baile",
    "no passinho", "tremedeira", "pancadão", "paredão", "grave",
    "grave batendo", "tremendo", "estourado", "estourada", "chavosa",
    "chavoso", "chavão", "chavinha", "chave", "chavezinha", "marra",
    "marrenta", "marrento", "postura", "presença", "respeito", "visão",
    "visão de cria", "sangue bom", "sinistro", "pesadão", "pesadona",
    "brabo", "braba", "brabíssimo", "nervoso", "nervosinha", "avançada",
    "avançado", "envolvente", "envolvência", "malemolência", "sedução",
    "tentação", "proibidão", "pescoço duro", "bolado", "boladona",
    "boladão", "desenrolado", "desenrolada", "fechamento", "fechado",
    "fortão", "fortinha", "puta", "calcinha","piru", "piroca", "crime", "gostosa", "vagaba", "putaria",
    "porra","caralho","puta","putaria","fuder","foda","merda","cuzão","cuzao","buceta","piranha","vadia", "putaria", "safada", "bandida",
    "macho", "maxo", "cachorra", "cachorro", "oitao", "otario", "bereta", "os hommi", "comeca a botar", "sentando", "de ladinho", "safada",
    "safado", "raba", "rabao", "chupa", "chupada", "chupando", "chupa chupa",
    "cu", "sentar", "quicar", "rebolar",
    "descer", "foder", "gemer", "botar", "dar", "safada", "putaria",
    "tesão", "gostosa", "bundão", "rabão", "novinha", "piranha",
    "cachorra", "vagabunda", "filho da puta", "otário", "trouxa",
    "corno", "pau no cu", "arrombado", "babaca", "patrão", "chefe",
    "bandida", "maloka", "cria", "mandrake", "vida loka", "postura",
    "presença", "respeito", "baile", "paredão", "pancadão", "grave",
    "grave batendo", "tremendo", "rajada", "estourado", "fluxo",
    "passinho", "B.O.", "glock", "blindado", "envolvido", "fita",
    "proceder", "responsa", "quadrilha", "disposição", "no corre",
    "chavosa", "chavoso", "marra", "marrenta", "pesadão", "brabo",
    "sinistro", "nervoso", "visão", "visão de cria", "no grau",
    "na maldade", "no pique", "no talento", "no fluxo", "no baile",
    "fogo", "malemolência", "sedução", "tentação", "vem", "vai",
    "joga", "bota", "sobe", "desce", "encosta", "toma", "vira",
    "chama", "agora", "hoje", "sempre", "nunca", "tudo", "nada",
    "muito", "tudo nosso", "favela", "quebrada", "sarrar", "sarração", "encaixar", "encaixe", "jogada", "mexer",
    "mexida", "mexe", "chacoalhar", "balançar", "rebolada", "sentada",
    "quicada", "travada", "empinar", "empinada", "deslizar", "escorregar",
    "colada", "coladinha", "pele", "suor", "calor", "pegada", "pegador",
    "pegadora", "malícia", "safadeza pura", "sem limite", "sem freio",
    "sem censura", "sem pudor", "ousada", "ousado", "atrevimento total",
    "liberdade", "libertina", "libertino", "luxúria", "prazer",
    "prazerosa", "prazeroso", "excitação", "ardente", "intenso",
    "intensa", "provoca", "provocação", "dominar", "dominante",
    "dominada", "controle", "sem controle", "perigo", "tentador",
    "tentadora", "instinto", "impulso", "clima", "climinha",
    "vibe pesada", "vibe quente", "pressão", "pressão total",
    "impacto", "choque", "batida seca", "grave pesado", "subgrave",
    "sub pesado", "batendo forte", "estrondo", "explodir", "explosão",
    "quente demais", "suando", "suadinha", "energia", "energia bruta",
    "presença forte", "domínio", "liderança", "moral", "conceito",
    "conceituada", "conceituado", "estilo próprio", "postura firme", "mandela", "mandelao",
    "mandelinha", "autenticado", "paga de grandao", "chupa chupa", "chupa-chupa", "chupachupa",
    "chup", "pipokinha", "pipoquinha", "espancamento", "sessao de espancamento", "espancar",
    "gay", "nazi", "facismo", "fascist", "facista", "fascista", "minoria", "minority", "demonio", "demon", "devil","demônio",
    "estupram", "estrupar", "estrupamento", "homofóbico"


])

SELF_HARM_WORDS = set([
    "suicide","kill myself","end my life","self harm","self-harm","cut myself","cutting",
    "hang myself","overdose","die tonight","want to die","no reason to live"
    "pussy", "dick", "cock", "shaft", "prick", "cunt", "snatch", "asshole",
    "asshole", "fuck", "sit on it", "twerk", "go down", "bounce",
    "put it in", "give it", "thrust", "moan", "slap", "slut",
    "son of a bitch", "hoe", "slut", "bitch", "jerk", "asshole",
    "sucker", "cuckold", "stick it up your ass", "asshole (insult)",
    "hood", "street hustler", "street smart", "kid from the hood",
    "the favela won", "kid", "boss", "crazy life", "chief",
    "bad girl", "young girl", "police report", "armored",
    "glock", "spray shooting", "involved", "scheme",
    "code of conduct", "gang", "responsibility", "ready for anything",
    "lewdness", "fuckery", "manwhore", "slutty woman", "pervert",
    "perverted woman", "horny", "hot girl", "sexy girl", "hot guy",
    "big ass", "fat ass", "thick girl", "big tits", "big boobs",
    "naughtiness", "sleazy", "sleazy woman", "boldness",
    "provocative", "shameless woman", "shameless man",
    "no shame", "bold girl", "bold behavior", "fire",
    "on point", "perfect timing", "dirty mindset",
    "natural talent", "in the vibe", "on the grind",
    "in the flow", "at the party", "dance moves",
    "shaking", "bass blast", "wall of speakers",
    "bass", "bass hitting", "vibrating",
    "blown up", "popping off", "stylish girl",
    "stylish guy", "big style", "little style",
    "key", "little key", "swagger",
    "bossy woman", "bossy man", "posture",
    "presence", "respect", "vision",
    "hood vision", "good vibes", "grimy",
    "heavy hitter", "heavy vibe",
    "badass", "badass woman", "ultimate badass",
    "angry", "feisty girl", "advanced woman",
    "advanced man", "engaging", "engagement",
    "groove", "seduction", "temptation",
    "forbidden funk", "stiff neck",
    "stressed out", "very stressed",
    "super stressed", "smooth talker",
    "smooth talker woman", "closure",
    "closed deal", "strong guy", "strong girl" "Buceta", "piroca", "pau", "rola", "pica", "xoxota", "xana", "cu",
    "arrombado", "foder", "sentar", "rebolar", "descer", "quicar", "botar",
    "dar", "socar", "gemer", "tapão", "safada", "filho da puta",
    "vagabunda", "piranha", "cachorra", "otário", "babaca", "trouxa",
    "corno", "pau no cu", "arrombado (insulto)", "maloka", "maloqueiro",
    "mandrake", "cria", "favela venceu", "menor", "patrão", "vida loka",
    "chefe", "bandida", "novinha", "B.O.", "blindado", "Glock", "rajada",
    "envolvido", "fita", "proceder", "quadrilha", "responsa", "disposição",
    "Safadeza", "putaria", "putão", "putona", "tarado", "tarada", "tesão",
    "tesuda", "gostosa", "gostoso", "rabão", "bundão", "bunduda", "peitão",
    "peituda", "maldade", "malicioso", "maliciosa", "ousadia",
    "provocante", "descarada", "descarado", "sem vergonha", "atrevida",
    "atrevimento", "fogo", "no grau", "no ponto", "na maldade",
    "no talento", "no pique", "no corre", "no fluxo", "no baile",
    "no passinho", "tremedeira", "pancadão", "paredão", "grave",
    "grave batendo", "tremendo", "estourado", "estourada", "chavosa",
    "chavoso", "chavão", "chavinha", "chave", "chavezinha", "marra",
    "marrenta", "marrento", "postura", "presença", "respeito", "visão",
    "visão de cria", "sangue bom", "sinistro", "pesadão", "pesadona",
    "brabo", "braba", "brabíssimo", "nervoso", "nervosinha", "avançada",
    "avançado", "envolvente", "envolvência", "malemolência", "sedução",
    "tentação", "proibidão", "pescoço duro", "bolado", "boladona",
    "boladão", "desenrolado", "desenrolada", "fechamento", "fechado",
    "fortão", "fortinha", "putaria", "porra","caralho","puta","putaria","fuder","foda","merda","cuzão","cuzao","buceta","piranha","vadia", "putaria", "safada", "bandida",
    "macho", "maxo", "cachorra", "cachorro", "oitao", "otario", "bereta", "os hommi", "comeca a botar", "sentando", "de ladinho", "safada",
    "safado", "raba", "rabao", "chupa", "chupada", "chupando", "chupa chupa",
    "cu", "sentar", "quicar", "rebolar",
    "descer", "foder", "gemer", "botar", "dar", "safada", "putaria",
    "tesão", "gostosa", "bundão", "rabão", "novinha", "piranha",
    "cachorra", "vagabunda", "filho da puta", "otário", "trouxa",
    "corno", "pau no cu", "arrombado", "babaca", "patrão", "chefe",
    "bandida", "maloka", "cria", "mandrake", "vida loka", "postura",
    "presença", "respeito", "baile", "paredão", "pancadão", "grave",
    "grave batendo", "tremendo", "rajada", "estourado", "fluxo",
    "passinho", "B.O.", "glock", "blindado", "envolvido", "fita",
    "proceder", "responsa", "quadrilha", "disposição", "no corre",
    "chavosa", "chavoso", "marra", "marrenta", "pesadão", "brabo",
    "sinistro", "nervoso", "visão", "visão de cria", "no grau",
    "na maldade", "no pique", "no talento", "no fluxo", "no baile",
    "fogo", "malemolência", "sedução", "tentação", "vem", "vai",
    "joga", "bota", "sobe", "desce", "encosta", "toma", "vira",
    "chama", "agora", "hoje", "sempre", "nunca", "tudo", "nada",
    "muito", "tudo nosso", "favela", "quebrada", "sarrar", "sarração", "encaixar", "encaixe", "jogada", "mexer",
    "mexida", "mexe", "chacoalhar", "balançar", "rebolada", "sentada",
    "quicada", "travada", "empinar", "empinada", "deslizar", "escorregar",
    "colada", "coladinha", "pele", "suor", "calor", "pegada", "pegador",
    "pegadora", "malícia", "safadeza pura", "sem limite", "sem freio",
    "sem censura", "sem pudor", "ousada", "ousado", "atrevimento total",
    "liberdade", "libertina", "libertino", "luxúria", "prazer",
    "prazerosa", "prazeroso", "excitação", "ardente", "intenso",
    "intensa", "provoca", "provocação", "dominar", "dominante",
    "dominada", "controle", "sem controle", "perigo", "tentador",
    "tentadora", "instinto", "impulso", "clima", "climinha",
    "vibe pesada", "vibe quente", "pressão", "pressão total",
    "impacto", "choque", "batida seca", "grave pesado", "subgrave",
    "sub pesado", "batendo forte", "estrondo", "explodir", "explosão",
    "quente demais", "suando", "suadinha", "energia", "energia bruta",
    "presença forte", "domínio", "liderança", "moral", "conceito",
    "conceituada", "conceituado", "estilo próprio", "postura firme", "mandela", "mandelao",
    "mandelinha", "autenticado", "paga de grandao", "chupa chupa", "chupa-chupa", "chupachupa",
    "chup", "pipokinha", "pipoquinha", "espancamento", "sessao de espancamento", "espancar",
    "gay", "nazi", "facismo", "fascist", "facista", "fascista", "minoria", "minority", "demonio", "demon", "devil","demônio",
    "estupram", "estrupar", "estrupamento", "homofóbico"



])

CRIME_WORDS = set([
    "rob","robbery","steal","stole","theft","fraud","scam","gang","dealer","dealers",
    "trap","trappin","gunman","hitman","extortion","kidnap","kidnapping", "gay"
])


EXPLICIT_WORDS = set([
    # English
    "fuck","fucking","shit","bitch","asshole","motherfucker","cunt","dick","pussy", "gay"
    # Portuguese (BR)
    "porra","caralho","puta","putaria","fuder","foda","merda","cuzão","cuzao","buceta","piranha","vadia", "putaria", "safada", "bandida",
    "macho", "maxo", "cachorra", "cachorro", "oitao", "otario", "bereta", "os hommi", "comeca a botar", "sentando", "de ladinho", "safada",
    "safado", "raba", "rabao", "chupa", "chupada", "chupando", "chupa chupa",
    "buceta", "pau", "pica", "rola", "cu", "sentar", "quicar", "rebolar",
    "descer", "foder", "gemer", "botar", "dar", "safada", "putaria",
    "tesão", "gostosa", "bundão", "rabão", "novinha", "piranha",
    "cachorra", "vagabunda", "filho da puta", "otário", "trouxa",
    "corno", "pau no cu", "arrombado", "babaca", "patrão", "chefe",
    "bandida", "maloka", "cria", "mandrake", "vida loka", "postura",
    "presença", "respeito", "baile", "paredão", "pancadão", "grave",
    "grave batendo", "tremendo", "rajada", "estourado", "fluxo",
    "passinho", "B.O.", "glock", "blindado", "envolvido", "fita",
    "proceder", "responsa", "quadrilha", "disposição", "no corre",
    "chavosa", "chavoso", "marra", "marrenta", "pesadão", "brabo",
    "sinistro", "nervoso", "visão", "visão de cria", "no grau",
    "na maldade", "no pique", "no talento", "no fluxo", "no baile",
    "fogo", "malemolência", "sedução", "tentação", "vem", "vai",
    "joga", "bota", "sobe", "desce", "encosta", "toma", "vira",
    "chama", "agora", "hoje", "sempre", "nunca", "tudo", "nada",
    "muito", "tudo nosso", "favela", "quebrada", "sarrar", "sarração", "encaixar", "encaixe", "jogada", "mexer",
    "mexida", "mexe", "chacoalhar", "balançar", "rebolada", "sentada",
    "quicada", "travada", "empinar", "empinada", "deslizar", "escorregar",
    "colada", "coladinha", "pele", "suor", "calor", "pegada", "pegador",
    "pegadora", "malícia", "safadeza pura", "sem limite", "sem freio",
    "sem censura", "sem pudor", "ousada", "ousado", "atrevimento total",
    "liberdade", "libertina", "libertino", "luxúria", "prazer",
    "prazerosa", "prazeroso", "excitação", "ardente", "intenso",
    "intensa", "provoca", "provocação", "dominar", "dominante",
    "dominada", "controle", "sem controle", "perigo", "tentador",
    "tentadora", "instinto", "impulso", "clima", "climinha",
    "vibe pesada", "vibe quente", "pressão", "pressão total",
    "impacto", "choque", "batida seca", "grave pesado", "subgrave",
    "sub pesado", "batendo forte", "estrondo", "explodir", "explosão",
    "quente demais", "suando", "suadinha", "energia", "energia bruta",
    "presença forte", "domínio", "liderança", "moral", "conceito",
    "conceituada", "conceituado", "estilo próprio", "postura firme", "mandela", "mandelao",
    "mandelinha", "autenticado", "paga de grandao", "chupa chupa", "chupa-chupa", "chupachupa",
    "chup", "pipokinha", "pipoquinha", "espancamento", "sessao de espancamento", "espancar",
    "gay", "nazi", "facismo", "fascist", "facista", "fascista", "minoria", "minority", "demonio", "demon", "devil","demônio",
    "estupram", "estrupar", "estrupamento"



])
# ======================================================
# MODEL CACHING (stable)
# ======================================================

@lru_cache(maxsize=2)
def get_whisper(size="base"):
    return whisper.load_model(size)

@lru_cache(maxsize=1)
def get_sentiment():
    tok = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    mdl = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")
    return tok, mdl

def roberta_sentiment(text: str):
    tok, mdl = get_sentiment()
    t = (text or "")[:1200]
    inputs = tok(t, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = mdl(**inputs).logits
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return {"negative": float(probs[0]), "neutral": float(probs[1]), "positive": float(probs[2])}

# ======================================================
# HELPERS
# ======================================================

def token_counts(text: str, vocab: set[str]) -> int:
    if not text:
        return 0
    toks = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for t in toks if t in vocab)

def sample_audio_for_fft(y, sr):
    win = int(20 * sr)
    if len(y) <= win:
        return y
    return np.concatenate([y[:win], y[len(y)//2:len(y)//2+win], y[-win:]])

def pretty_duration(seconds: float) -> str:
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m}m {s:02d}s"

def audio_fingerprint(y, sr):
    # stable small fingerprint (for caching) from start/mid/end
    y = np.asarray(y, dtype=np.float32)
    n = len(y)
    if n == 0:
        return "empty"
    take = min(n, sr*10)
    a = y[:take]
    b = y[n//2:n//2+take] if n//2+take <= n else y[n//2:]
    c = y[-take:]
    blob = np.concatenate([a, b, c])
    h = hashlib.sha1(blob.tobytes() + str(sr).encode()).hexdigest()
    return h

# ======================================================
# AUDIO SAFETY SIGNALS (noise / harshness / piercing tone / extremes)
# ======================================================

def compute_audio_safety(y, sr, fast_mode: bool):
    y_eval = sample_audio_for_fft(y, sr) if fast_mode else y

    rms = float(np.sqrt(np.mean(y_eval**2)) + 1e-12)
    rms_db = float(20*np.log10(rms + 1e-12))

    fft = np.abs(np.fft.rfft(y_eval))
    freqs = np.fft.rfftfreq(len(y_eval), 1/sr)
    total = float(np.sum(fft) + 1e-12)

    infra_ratio = float(np.sum(fft[freqs < 20]) / total) if len(freqs) else 0.0
    ultra_ratio = float(np.sum(fft[freqs > 18000]) / total) if len(freqs) else 0.0

    crest = float((np.max(fft) + 1e-12) / (np.mean(fft) + 1e-12))

    hop = 512
    n_fft = 2048
    centroid = librosa.feature.spectral_centroid(y=y_eval, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y_eval, sr=sr, n_fft=n_fft, hop_length=hop, roll_percent=0.85)[0]
    flatness = librosa.feature.spectral_flatness(y=y_eval, n_fft=n_fft, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_eval, frame_length=n_fft, hop_length=hop)[0]
    rms_f = librosa.feature.rms(y=y_eval, frame_length=n_fft, hop_length=hop)[0]

    def q(x, p):
        return float(np.quantile(x, p)) if len(x) else 0.0

    c_med, c_p95 = q(centroid, 0.5), q(centroid, 0.95)
    r_p95 = q(rolloff, 0.95)
    f_med = q(flatness, 0.5)
    z_med = q(zcr, 0.5)
    rms_med, rms_p95 = q(rms_f, 0.5), q(rms_f, 0.95)

    noise_like = (f_med > 0.35 and z_med > 0.08)
    piercing_tone = (c_med > 3800 and f_med < 0.18 and crest > 12.0)
    harsh_bright = (c_p95 > 6500 or r_p95 > 12000) and (f_med > 0.22 or z_med > 0.06)
    transient_spiky = (rms_med > 1e-9) and ((rms_p95 / rms_med) > 3.0)
    loudish = (rms_db > -18.0)

    points = 0
    flags = []

    if infra_ratio > 0.08:
        points += 4
        flags.append("Noticeable infrasonic energy (<20 Hz) — can feel uncomfortable at volume")
    if ultra_ratio > 0.10:
        points += 3
        flags.append("Strong very-high-frequency energy (>18 kHz) — can feel irritating/fatiguing")

    if noise_like:
        points += 8
        flags.append("Noise-like spectrum (broadband) — typical of traffic/city noise; can be tiring")
    if piercing_tone:
        points += 10
        flags.append("Piercing tonal component (narrow high tone) — can feel painful/irritating")
    if harsh_bright:
        points += 6
        flags.append("Harsh/bright profile (high centroid/rolloff) — fatigue risk")
    if transient_spiky and loudish:
        points += 4
        flags.append("Sharp transients at higher level — startle/fatigue risk")

    if noise_like and not piercing_tone:
        sound_type = "Noise-like / ambient"
    elif piercing_tone:
        sound_type = "Tonal / piercing"
    else:
        sound_type = "Music-like / tonal"

    hard_not = bool(piercing_tone or (noise_like and harsh_bright and loudish))

    return {
        "points": int(points),
        "hard_not": hard_not,
        "sound_type": sound_type,
        "flags": flags[:6],
        "metrics": {
            "rms_db": rms_db,
            "centroid_med": c_med,
            "rolloff_p95": r_p95,
            "flatness_med": f_med,
            "zcr_med": z_med,
            "crest": crest,
            "infra_ratio": infra_ratio,
            "ultra_ratio": ultra_ratio
        }
    }

# ======================================================
# SMART VOCAL / SPEECH GATING (to skip Whisper when pointless)
# ======================================================

def likely_has_vocals(y, sr):
    # Cheap heuristic: music-like + not too flat + centroid in voice-ish region
    # (Not perfect, but enough to avoid wasting time on pure noise)
    y_s = sample_audio_for_fft(y, sr)
    hop = 512
    n_fft = 2048
    flat = librosa.feature.spectral_flatness(y=y_s, n_fft=n_fft, hop_length=hop)[0]
    cent = librosa.feature.spectral_centroid(y=y_s, sr=sr, n_fft=n_fft, hop_length=hop)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y_s, frame_length=n_fft, hop_length=hop)[0]
    f_med = float(np.quantile(flat, 0.5))
    c_med = float(np.quantile(cent, 0.5))
    z_med = float(np.quantile(zcr, 0.5))
    # voice-ish centroid often ~800–3500 in mixed music; noise tends to be flatter + higher zcr
    return (f_med < 0.45) and (500 <= c_med <= 6000) and (z_med < 0.20)

# ======================================================
# OPTION A — SEGMENT TRANSCRIPTION (anchor segments)
# start / mid / end + loudest window
# ======================================================

def pick_anchor_segments(y, sr, duration, seg_len=18.0, hop_s=2.0):
    # 3 fixed anchors
    anchors = []
    if duration <= seg_len + 2:
        anchors.append(0.0)
        return anchors

    anchors.append(min(10.0, max(0.0, duration - seg_len))) # near start
    anchors.append(max(0.0, duration/2 - seg_len/2)) # middle
    anchors.append(max(0.0, duration - seg_len - 2.0)) # near end

    # loudest segment (RMS over sliding windows)
    step = int(hop_s * sr)
    win = int(seg_len * sr)
    if len(y) > win + step:
        rms_vals = []
        starts = []
        for s in range(0, len(y) - win, step):
            seg = y[s:s+win]
            rms = float(np.sqrt(np.mean(seg**2)) + 1e-12)
            rms_vals.append(rms)
            starts.append(s)
        if rms_vals:
            best = starts[int(np.argmax(rms_vals))]
            anchors.append(best / sr)

    # deduplicate and clamp
    cleaned = []
    for t in anchors:
        t = float(max(0.0, min(duration - seg_len, t)))
        if all(abs(t - u) > 3.0 for u in cleaned):
            cleaned.append(t)
    return cleaned[:4] # up to 4 segments

def write_segment_wav(y, sr, start_s, seg_len, tmpdir):
    start = int(start_s * sr)
    end = int(min(len(y), start + int(seg_len * sr)))
    seg = y[start:end]

    # Whisper works best at 16k mono
    target_sr = 16000
    if sr != target_sr:
        seg = librosa.resample(seg, orig_sr=sr, target_sr=target_sr)
        sr2 = target_sr
    else:
        sr2 = sr

    path = os.path.join(tmpdir, f"seg_{int(start_s*1000)}.wav")
    sf.write(path, seg, sr2)
    return path

def transcribe_anchor_segments(y, sr, audio_path, fast_mode):
    duration = len(y)/sr
    seg_len = 14.0 if fast_mode else 18.0
    anchors = pick_anchor_segments(y, sr, duration, seg_len=seg_len)

    # if too short, just transcribe whole (rare)
    model = get_whisper("tiny" if fast_mode else "base")

    tmpdir = tempfile.mkdtemp(prefix="segs_")
    try:
        texts = []
        for t in anchors:
            p = write_segment_wav(y, sr, t, seg_len, tmpdir)
            out = model.transcribe(p, fp16=torch.cuda.is_available())
            txt = (out.get("text") or "").strip()
            if txt:
                texts.append(txt)
        return " ".join(texts).lower().strip()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ======================================================
# YOUTUBE DOWNLOAD (no cookies UI; clean error hint)
# ======================================================

def download_youtube_audio(url: str, fast_mode: bool):
    tmpdir = tempfile.mkdtemp(prefix="yt_")
    outtmpl = os.path.join(tmpdir, "audio.%(ext)s")
    fmt = "bestaudio[abr<=96]/bestaudio" if fast_mode else "bestaudio/best"
    ydl_opts = {
        "format": fmt,
        "outtmpl": outtmpl,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "postprocessors": [{"key":"FFmpegExtractAudio","preferredcodec":"wav"}],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        title = info.get("title", "Unknown title")
        channel = info.get("uploader", "Unknown channel")

    wav_path = os.path.join(tmpdir, "audio.wav")
    if not os.path.exists(wav_path):
        raise RuntimeError("WAV not created. ffmpeg may be missing.")
    return wav_path, title, channel, tmpdir

# ======================================================
# GLOBAL CACHE (repeat runs become instant)
# ======================================================
ANALYSIS_CACHE = {}

# ======================================================
# CORE ANALYSIS
# ======================================================

def analyze_audio(audio_path, fast_mode, title, channel):
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    duration = len(y) / sr

    # cache key from waveform fingerprint + mode + title
    key = f"{audio_fingerprint(y, sr)}::{int(fast_mode)}::{title}"
    if key in ANALYSIS_CACHE:
        return ANALYSIS_CACHE[key]

    # FFT band profile (sampled in fast)
    y_fft = sample_audio_for_fft(y, sr) if fast_mode else y
    fft = np.abs(np.fft.rfft(y_fft))
    freqs = np.fft.rfftfreq(len(y_fft), 1/sr)

    band_energy = {}
    for b,(lo,hi) in BANDS.items():
        idx = np.where((freqs>=lo)&(freqs<hi))[0]
        band_energy[b] = float(np.sum(fft[idx]))
    total = sum(band_energy.values()) or 1.0
    profile = {b: float(band_energy[b]/total) for b in band_energy}

    # audio safety (fast uses sampled internally)
    audio_safety = compute_audio_safety(y, sr, fast_mode)

    # chart
    fig = plt.figure(figsize=(10,4))
    plt.bar(profile.keys(), profile.values(), color=[COLORS[b] for b in profile])
    plt.title(f"{title} — {channel}")
    plt.ylabel("Relative Energy")
    plt.tight_layout()

    # ====== Lyrics transcription (Option A) with gating ======
    lyrics = ""
    lang = "unknown"
    sent = {"negative": 0.0, "neutral": 1.0, "positive": 0.0}

    # if noise-like/piercing, skip lyrics for speed (usually no lyrics anyway)
    do_lyrics = (audio_safety["sound_type"] == "Music-like / tonal") and likely_has_vocals(y, sr)

    if do_lyrics:
        lyrics = transcribe_anchor_segments(y, sr, audio_path, fast_mode=fast_mode)
        if lyrics:
            try:
                lang = detect(lyrics)
            except LangDetectException:
                lang = "unknown"
            sent = roberta_sentiment(lyrics)

    # Lexicon scores (still computed, but NOT shown as "Detected Themes")
    scores = {
        "drugs": token_counts(lyrics, DRUG_WORDS),
        "violence": token_counts(lyrics, VIOLENCE_WORDS),
        "sexual": token_counts(lyrics, SEXUAL_WORDS),
        "selfharm": token_counts(lyrics, SELF_HARM_WORDS),
        "crime": token_counts(lyrics, CRIME_WORDS),
        "explicit": token_counts(lyrics, EXPLICIT_WORDS),
    }

    explicit_points = 0
    if scores.get('explicit', 0) >= 6:
        explicit_points = 12
    elif scores.get('explicit', 0) >= 3:
        explicit_points = 8
    elif scores.get('explicit', 0) >= 1:
        explicit_points = 4

    # Risk points (includes audio safety)
    risk_points = (
        scores["selfharm"]*12 +
        scores["drugs"]*4 +
        scores["violence"]*4 +
        scores["sexual"]*3 +
        scores["crime"]*4 +
        (sent["negative"]>0.75)*6 +
        audio_safety["points"] + explicit_points
    )

    # Verdict
    if scores["selfharm"] >= 1 or audio_safety["hard_not"] or risk_points >= 22:
        verdict = "NOT RECOMMENDED"
    elif risk_points >= 8:
        verdict = "USE WITH MODERATION"
    else:
        verdict = "RECOMMENDED"


    # If explicit language appears in lyrics, never label as RECOMMENDED (internal rule)
    if verdict == "RECOMMENDED" and scores.get('explicit', 0) >= 1:
        verdict = "USE WITH MODERATION"

    # USER OUTPUT (NO "Detected Themes")
    lines = []
    lines.append(f"# 🎵 {title}")
    lines.append(f"### Channel: {channel}")
    lines.append(f"## Verdict: **{verdict}**")

    lines.append("### Listening Context")
    lines.append(f"- **Mode:** {'FAST' if fast_mode else 'ACCURATE'}")
    lines.append(f"- **Length:** {pretty_duration(duration)}")
    lines.append(f"- **Sound type:** {audio_safety['sound_type']}")

    # explain sentiment in human language (only if lyrics used)
    if do_lyrics and lyrics:
        lines.append("### Lyric Emotional Signal (summary)")
        # translate internal numbers to plain language
        neg, neu, pos = sent["negative"], sent["neutral"], sent["positive"]
        if neg >= 0.65 and neg > pos:
            tone = "Strong negative emotional tone (anger, distress, bleakness)"
        elif pos >= 0.65 and pos > neg:
            tone = "Strong positive emotional tone (hopeful, uplifting, affectionate)"
        elif neu >= 0.55:
            tone = "Mostly neutral tone (descriptive, flat, or mixed without strong emotion)"
        else:
            tone = "Mixed emotional tone (push-pull language; not clearly one-sided)"
        lines.append(f"- **Tone:** {tone}")

    # Short reason
    lines.append("\n---\n## Frequency Insights")
    for b in BANDS:
        info = FREQ_HUMAN[b]
        pct = profile[b]*100
        lines.append(f"### {b} ({info['range']}) — {pct:.1f}%")

        if verdict == "RECOMMENDED":
            lines.append(f"- **Effect:** {info['effect']}")
            lines.append(f"- **Benefit:** {info['benefit']}")
        else:
            # For USE WITH MODERATION / NOT RECOMMENDED, show only negative-facing effects + risks (no benefits)
            eff = NEGATIVE_EFFECTS.get(b, info['effect'])
            lines.append(f"- **Effect:** {eff}")
            lines.append(f"- **Risk:** {info['risk']}")

    result = (fig, "\n".join(lines))
    ANALYSIS_CACHE[key] = result
    return result

# ======================================================
# GRADIO RUNNER
# ======================================================

def run(upload, yt, fast):
    tmp=None
    try:
        if yt and yt.strip():
            path, title, channel, tmp = download_youtube_audio(yt.strip(), fast)
        else:
            path = upload
            title = os.path.basename(upload) if upload else "Local file"
            channel = "Local upload"

        if not path:
            return None, "Please upload an audio file OR paste a YouTube link."

        return analyze_audio(path, fast, title, channel)

    except Exception as e:
        msg = str(e)
        # YouTube bot-check / auth
        if ("confirm you’re not a bot" in msg) or ("confirm you're not a bot" in msg) or ("sign in to confirm" in msg):
            hint = (
                "YouTube blocked automated downloads for this link.\n\n"
                "**Fix options:**\n"
                "1) Use **Upload audio** (works every time), or\n"
                "2) Try a different YouTube link (some links work, some don’t).\n\n"
                "This is a YouTube restriction, not your analysis code."
            )
            return None, f"❌ Error: {msg}\n\n{hint}\n\n```text\n{traceback.format_exc()}\n```"
        return None, f"❌ Error: {msg}\n\n```text\n{traceback.format_exc()}\n```"
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)

# ======================================================
# UI
# ======================================================

with gr.Blocks() as demo:
    gr.Markdown("# 🎵 Frequency Insight\nUpload audio or paste a YouTube link, then analyze.")
    up = gr.Audio(type="filepath", label="Upload audio (wav/mp3)")
    yt = gr.Textbox(label="YouTube link (optional)", placeholder="https://youtube.com/watch?v=...")
    # Removed fast mode checkbox

    btn = gr.Button("Analyze")
    plot = gr.Plot()
    text = gr.Markdown()

    # Changed btn.click to pass False for fast_mode by default
    btn.click(lambda upload_val, yt_val: run(upload_val, yt_val, False), [up, yt], [plot, text])

demo.queue().launch(server_name="0.0.0.0", server_port=int(os.getenv("PORT", "7860")), share=False)
