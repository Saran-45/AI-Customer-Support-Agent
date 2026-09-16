"""
Synthetic corpus generator.

IMPORTANT: This module exists ONLY because this build environment has no
network access to kaggle.com or huggingface.co (see DECISION_LOG.md,
Decision 1). It produces a *clearly synthetic* dataset for a fictional
brand ("StreamBoxHelp") that mimics the exact column schema of the real
`thoughtvector/customer-support-on-twitter` dataset (tweet_id, author_id,
inbound, created_at, text, response_tweet_id, in_response_to_tweet_id) so
that every downstream script (build_threads.py, intent discovery,
retrieval indexing, evaluation) runs unmodified against real data once
it's available.

Every "customer message" and "historical resolution" below is generated
from templates + randomized slot-filling -- it is NOT sampled from any
real corpus and must never be described as real customer data.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime, timedelta

import pandas as pd

BRAND_HANDLE = "StreamBoxHelp"

# (intent, customer message templates, agent resolution templates, base escalate rate)
INTENT_TEMPLATES: dict[str, dict] = {
    "login_issue": {
        "customer": [
            "I keep getting logged out of the app every time I open it {device}",
            "Your app won't let me sign in, keeps saying invalid credentials {device}",
            "Can't log in at all today, just spinning forever {device}",
            "Login screen just loops back to itself, tried {n} times {device}",
        ],
        "agent": [
            "Sorry about that! Please try updating to the latest app version and logging in again. If it persists DM us your email so we can check your account.",
            "That sounds frustrating. Could you try clearing the app cache and logging in again? Let us know if it keeps happening.",
        ],
        "escalate_rate": 0.15,
    },
    "password_reset": {
        "customer": [
            "The password reset email never arrives, checked spam too {device}",
            "How do I change my password on the app, can't find the option {device}",
            "Reset link says expired every single time {device}",
        ],
        "agent": [
            "You can request a new reset link from the login screen -- it can take up to 10 minutes to arrive. If it still doesn't show up, DM us the email on file.",
            "Please try the 'Forgot password' link again; if it keeps expiring, send us a DM with your account email and we'll help directly.",
        ],
        "escalate_rate": 0.10,
    },
    "payment_failed": {
        "customer": [
            "My card was charged but my subscription still shows as inactive {n} hours later",
            "Payment keeps failing even though my card is fine, tried {n} times",
            "Paid for premium but still seeing ads, what's going on",
        ],
        "agent": [
            "Sorry for the trouble! Payments can take up to 30 minutes to reflect. If it's still inactive after that, DM us your order number.",
            "We can look into that -- please DM your account email and the last 4 digits of the card used so we can check the transaction.",
        ],
        "escalate_rate": 0.30,
    },
    "billing_overcharge": {
        "customer": [
            "I was billed twice for the same month, please fix this",
            "You charged me after I cancelled last week, this isn't right",
            "Charged the annual price when I signed up for monthly, {n} dollars extra",
        ],
        "agent": [
            "That shouldn't happen -- please DM your account email and the charge date so our billing team can review and correct it.",
            "Apologies for the confusion, could you send your order ID via DM so we can look into the duplicate charge?",
        ],
        "escalate_rate": 0.55,
    },
    "subscription_cancel": {
        "customer": [
            "How do I cancel my subscription, can't find the button anywhere",
            "Please cancel my plan, I don't want to be charged again",
            "Trying to downgrade my plan but the app won't let me",
        ],
        "agent": [
            "You can cancel anytime from Account > Subscription > Cancel Plan. Let us know if you run into any issues!",
            "Sorry to see you go! Go to Settings > Subscription to cancel or downgrade. DM us if the option isn't showing.",
        ],
        "escalate_rate": 0.05,
    },
    "refund_request": {
        "customer": [
            "I want a refund for this month, the service didn't work at all",
            "Can I get my money back, I never used it after signing up",
            "This is unacceptable, I'm requesting a full refund for {n} months",
        ],
        "agent": [
            "We're sorry to hear that. Please DM your account email and order ID so our billing team can review your refund eligibility.",
            "Understood, we'll need to verify your account first -- DM us your email and we'll get this looked at.",
        ],
        "escalate_rate": 0.60,
    },
    "app_crash": {
        "customer": [
            "The app crashes every time I try to open it on my {device}",
            "It freezes on the loading screen, had to force close {n} times",
            "App crashes whenever a video starts playing on {device}",
        ],
        "agent": [
            "Sorry about that! Please make sure the app is updated to the latest version and try reinstalling. Let us know if the crash continues.",
            "That's not expected -- could you tell us your device model and app version so we can dig into this?",
        ],
        "escalate_rate": 0.20,
    },
    "playback_buffering": {
        "customer": [
            "Video keeps buffering even on good wifi, so annoying",
            "Quality drops to 240p randomly during playback on {device}",
            "Audio and video are out of sync for the last {n} days",
        ],
        "agent": [
            "Sorry for the disruption! Try lowering the streaming quality in Settings, or restart your router. Let us know if it continues.",
            "We'd suggest checking your connection speed and restarting the app. DM us if the buffering persists after that.",
        ],
        "escalate_rate": 0.10,
    },
    "content_unavailable": {
        "customer": [
            "Why was this show removed, I was in the middle of it",
            "It says not available in my country but I'm a paying subscriber",
            "The movie I was watching just disappeared from my list",
        ],
        "agent": [
            "Licensing agreements sometimes change and titles can be removed on short notice -- sorry for the inconvenience! We'll pass this along.",
            "Content availability varies by region due to licensing. We appreciate the feedback and will share it with the content team.",
        ],
        "escalate_rate": 0.05,
    },
    "account_security": {
        "customer": [
            "Someone is using my account, I see devices I don't recognize logged in",
            "My email was changed without my permission, please help",
            "I think my account got hacked, can't log in anymore",
        ],
        "agent": [
            "This needs immediate attention -- please DM us right away so our security team can lock down your account.",
            "We take this seriously. Please DM your account email immediately and avoid using the account until we confirm it's secure.",
        ],
        "escalate_rate": 0.95,
    },
    "feature_request": {
        "customer": [
            "You should add a way to download episodes for offline viewing",
            "Please add profiles for kids, would make this so much better",
            "Would love a dark mode option in the app",
        ],
        "agent": [
            "Thanks for the suggestion! We'll pass this along to our product team.",
            "Great idea, we've logged this as feedback for our roadmap team!",
        ],
        "escalate_rate": 0.02,
    },
    "positive_feedback": {
        "customer": [
            "Just wanted to say the new app update is great, thanks!",
            "Loving the new recommendations feature, keep it up",
            "Customer support just helped me out super fast, appreciated",
        ],
        "agent": [
            "Thank you so much for the kind words, we really appreciate it!",
            "That means a lot, thanks for being a subscriber!",
        ],
        "escalate_rate": 0.0,
    },
    "general_complaint": {
        "customer": [
            "Your customer support never responds, this is ridiculous",
            "Worst service I've ever used, so disappointed",
            "Been waiting {n} days for a reply, unbelievable",
        ],
        "agent": [
            "We're sorry for the delay and frustration -- DM us your account email so we can prioritize this.",
            "That's not the experience we want for you. Please DM us details so we can make it right.",
        ],
        "escalate_rate": 0.35,
    },
}

DEVICES = ["", "on my iPhone", "on Android", "on my smart TV", "on the web app", "on my iPad"]
NUMBERS = ["2", "3", "4", "5", "several"]


@dataclass
class _IdCounter:
    value: int = 1000000

    def next(self) -> int:
        self.value += 1
        return self.value


def _fill(template: str) -> str:
    return template.format(
        device=random.choice(DEVICES),
        n=random.choice(NUMBERS),
    ).replace("  ", " ").strip()


def generate_synthetic_dataset(n_conversations: int = 650, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic twcs-schema dataframe for the fictional brand.

    Mimics real-world messiness on purpose (see below) so preprocessing /
    leakage-prevention code has something real to handle:
      - ~4% of conversations have only the customer side (agent never
        replied in the window captured)
      - ~2% duplicate customer messages (customer double-posts)
      - a handful of records inserted slightly out of timestamp order
    """
    random.seed(seed)
    ids = _IdCounter()
    rows: list[dict] = []

    start = datetime(2024, 1, 1)
    intents = list(INTENT_TEMPLATES.keys())
    # Non-uniform popularity so the class distribution is imbalanced, like real data.
    weights = [14, 6, 9, 8, 10, 7, 9, 11, 5, 4, 4, 6, 7]

    for i in range(n_conversations):
        intent = random.choices(intents, weights=weights, k=1)[0]
        spec = INTENT_TEMPLATES[intent]
        conv_time = start + timedelta(hours=i * 6 + random.randint(0, 4))

        cust_id = f"cust_{random.randint(10000, 99999)}"
        cust_tweet_id = ids.next()
        cust_text = _fill(random.choice(spec["customer"]))

        rows.append({
            "tweet_id": cust_tweet_id,
            "author_id": cust_id,
            "inbound": True,
            "created_at": conv_time.isoformat(),
            "text": cust_text,
            "response_tweet_id": None,  # filled below if agent replies
            "in_response_to_tweet_id": None,
            "_intent_label": intent,  # ground-truth label, NOT part of real schema; stripped before saving raw
        })

        # ~4% one-sided conversations (no agent reply captured)
        if random.random() < 0.04:
            continue

        agent_tweet_id = ids.next()
        agent_text = random.choice(spec["agent"])
        reply_delay = timedelta(minutes=random.randint(5, 240))
        rows.append({
            "tweet_id": agent_tweet_id,
            "author_id": BRAND_HANDLE,
            "inbound": False,
            "created_at": (conv_time + reply_delay).isoformat(),
            "text": agent_text,
            "response_tweet_id": None,
            "in_response_to_tweet_id": cust_tweet_id,
            "_intent_label": intent,
        })
        rows[-2]["response_tweet_id"] = str(agent_tweet_id)

        # ~2% duplicate customer double-post
        if random.random() < 0.02:
            dup = dict(rows[-2])
            dup["tweet_id"] = ids.next()
            rows.append(dup)

        # ~25% get a short customer follow-up + second agent reply
        if random.random() < 0.25:
            followup_time = conv_time + reply_delay + timedelta(minutes=random.randint(2, 60))
            fu_tweet_id = ids.next()
            rows.append({
                "tweet_id": fu_tweet_id,
                "author_id": cust_id,
                "inbound": True,
                "created_at": followup_time.isoformat(),
                "text": random.choice(["Thanks, that worked!", "Still not working for me.", "Ok will try that.", "DMed you now."]),
                "response_tweet_id": None,
                "in_response_to_tweet_id": str(agent_tweet_id),
                "_intent_label": intent,
            })

    df = pd.DataFrame(rows)
    # Introduce a few out-of-order rows to exercise sorting logic downstream.
    shuffle_idx = df.sample(frac=0.03, random_state=seed).index
    df = pd.concat([df.drop(shuffle_idx), df.loc[shuffle_idx]])
    df = df.reset_index(drop=True)
    return df
