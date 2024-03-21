from semantic_router import Route

# we could use this as a guide for our chatbot to avoid political conversations
politics = Route(
    name="politics",
    utterances=[
        "isn't politics the best thing ever",
        "why don't you tell me about your political opinions",
        "don't you just love the president" "don't you just hate the president",
        "they're going to destroy this country!",
        "they will save the country!",
    ],
)

# this could be used as an indicator to our chatbot to switch to a more
# conversational prompt
chitchat = Route(
    name="chitchat",
    utterances=[
        "how's the weather today?",
        "how are things going?",
        "lovely weather today",
        "the weather is horrendous",
        "let's go to the chippy",
    ],
)

# we place both of our decisions together into single list
routes = [politics, chitchat]

import os
from semantic_router.encoders import CohereEncoder, OpenAIEncoder

# for Cohere
os.environ["COHERE_API_KEY"] = "<YOUR_API_KEY>"
encoder = CohereEncoder()

# or for OpenAI
os.environ["OPENAI_API_KEY"] = "<YOUR_API_KEY>"
encoder = OpenAIEncoder()


from semantic_router.layer import RouteLayer

rl = RouteLayer(encoder=encoder, routes=routes)

rl("don't you love politics?").name

rl("how's the weather today?").name

rl("I'm interested in learning about llama 2").name


# politics = Route(
#     name="politics",
#     utterances=[
#         "isn't politics the best thing ever",
#         "why don't you tell me about your political opinions",
#         "don't you just love the president" "don't you just hate the president",
#         "they're going to destroy this country!",
#         "they will save the country!",
#     ],
# )
#

interruption = Route(
                name="interruption",
            utterances=
                ["Sorry to interrupt", "Wait, no, that's not how things are",
                "Hold on right there, there is something else",
                "Excuse me for a moment",
"I hate to cut in, but",
"Pardon the interruption, but",
"Hang on, let me interject",
"Forgive me, but I need to interpose",
"I don't mean to disrupt, but",
"Allow me to break in for a second",
"Hold up, there's something I'd like to mention",
"If I may interject for a moment",
"I hope you don't mind me jumping in, but",
"Before you continue, there's one thing I'd like to add",
"Before we proceed, there's a point I'd like to make",
"I hate to interrupt, but there's something important to address",
"Can I just interject here for a quick comment?",
"Before we move forward, I'd like to bring up something"])

agreement = Route(
                name="agreement",
            utterances=["I agree", "Totally agreeing", "That's right",
             "I concur.",
"Absolutely.",
"I couldn't agree more.",
"Precisely.",
"You're spot on.",
"I'm in complete agreement.",
"Exactly.",
"You've hit the nail on the head.",
"I'm on the same page.",
"You're absolutely right.",
"I share the same view.",
"You've expressed it perfectly.",
"That's my sentiment exactly.",
"No doubt about it.",
"I'm with you on that."])

disagreement = Route(
                name="disagreement",
            utterances=["I disagree", "Totally disagreeing", "That's wrong", "I am not sure "
                "why you think this way",
"I beg to differ.",
"I don't see it that way.",
"I have a different perspective.",
"I respectfully disagree.",
"I'm afraid I can't agree with that.",
"I'm not entirely convinced.",
"I'm not on board with that idea.",
"I'm not sure I agree.",
"That doesn't resonate with me.",
"I'm not inclined to see it that way.",
"I'm not convinced of the validity.",
"I have reservations about that.",
"I'm not sold on that argument.",
"I have my doubts.",
"I see where you're coming from, but I have a different take."])

# Requests / demands markers
requests = Route(
                name="requests",
            utterances=["I want", "I demand", "I request", "I wish", "I'd like",
"Can I have",
"Could you please",
"I'm asking for",
"Would it be possible to",
"May I request",
"Is it okay if I",
"I wonder if you could",
"It would be great if",
"I was hoping for",
"Do you mind",
"I'd appreciate it if",
"Could you do me a favor and",
"If you don't mind, can you",
"I'm hoping you could help me with"])


questions = Route(
                name="question",
            utterances=["What do you think?", "Let me know what you think",
             "Tell me how you see this", "What are your thoughts?",
             "How do you feel about this?",
"Can you share your perspective on this?",
"I'd love to hear your opinion.",
"What's your take on this?",
"Do you have any thoughts on the matter?",
"What's your viewpoint?",
"I'm interested in your perspective.",
"How would you approach this?",
"I'm curious about your thoughts.",
"Can you give me your insight?",
"What's your impression of this situation?",
"I'm open to your ideas.",
"I value your opinion—what do you think?",
"I'm eager to hear your viewpoint.",
"How do you see things unfolding?"])

affirmations = Route(
                name="affirmation",
            utterances=["I state that", "I affirm that", "I strongly believe",
                "I think that", "My opinion is that", "My stance is",
"I declare that",
"I assert that",
"I am of the firm belief that",
"It is my conviction that",
"I hold the view that",
"I maintain that",
"I contend that",
"I posit that",
"I am confident that",
"I firmly state that",
"I insist that",
"I am convinced that",
"It is my assertion that",
"I am of the opinion that",
"I affirm without reservation that"])


"""
inspire from Inkitt and what they did in order to analyse reading behavior

for this scenario to analyse negotiation behavior

"""



