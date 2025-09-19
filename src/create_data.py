import pandas as pd
data = {
    "text": [
        "Breaking: Scientists discover water on Mars and confirm potential for life.",
        "Celebrity claims drinking bleach cures COVID-19.",
        "Government launches new initiative to support renewable energy adoption.",
        "Aliens landed in New York last night, according to anonymous sources.",
        "Stock markets rise as tech companies report record profits.",
        "Eating chocolate daily will make you live 200 years, researchers say.",
        "New vaccine approved after successful clinical trials.",
        "World will end next week because of asteroid impact, says viral WhatsApp message."
    ],
    "label": ["real","fake","real","fake","real","fake","real","fake"]
}
df = pd.DataFrame(data)
df.to_csv("data/sample_fake_news.csv", index=False)
print("Saved -> data/sample_fake_news.csv")
