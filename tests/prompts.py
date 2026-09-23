"""Fixed prompts for the correctness gates: varied domains and scripts, each at least 100 tokens
(the tests assert this), including multi-byte text so detokenization is exercised too."""

PROMPTS = [
    # 1. history prose
    "The history of the printing press is usually told as the story of a single inventor, but the "
    "reality is more tangled. Block printing had been practiced in East Asia for centuries, and movable "
    "type made of ceramic and later metal appeared in China and Korea long before it reached Europe. "
    "What changed in fifteenth-century Mainz was the combination of an oil-based ink, a screw press "
    "adapted from wine making, and a hand mould that could cast letters quickly, uniformly, and in "
    "very large numbers. Together these made printing cheap enough that",
    # 2. science explanation
    "Photosynthesis happens in two stages. In the light-dependent reactions, which take place in the "
    "thylakoid membranes of the chloroplast, pigments such as chlorophyll absorb photons and use their "
    "energy to split water molecules, releasing oxygen as a by-product. The energy captured is stored "
    "temporarily in two carrier molecules, ATP and NADPH. In the second stage, often called the Calvin "
    "cycle, the plant uses that stored energy to fix carbon dioxide from the air into three-carbon "
    "sugars. The enzyme responsible for the first step of carbon fixation is",
    # 3. Python code
    "import heapq\nfrom dataclasses import dataclass, field\n\n\n@dataclass(order=True)\nclass Job:\n"
    "    priority: int\n    name: str = field(compare=False)\n    duration: float = field(compare=False)\n\n\n"
    "class Scheduler:\n    \"\"\"Run jobs in priority order, lowest number first.\"\"\"\n\n    def __init__(self):\n"
    "        self._heap: list[Job] = []\n        self.completed: list[str] = []\n\n"
    "    def submit(self, job: Job) -> None:\n        heapq.heappush(self._heap, job)\n\n"
    "    def run_all(self) -> float:\n        \"\"\"Run every queued job and return the total time spent.\"\"\"\n",
    # 4. JavaScript code
    "// Fetch a list of users, retrying transient failures with exponential backoff.\n"
    "async function fetchWithRetry(url, { retries = 3, baseDelayMs = 200 } = {}) {\n"
    "  for (let attempt = 0; attempt <= retries; attempt++) {\n    try {\n"
    "      const response = await fetch(url);\n      if (response.status >= 500) {\n"
    "        throw new Error(`server error ${response.status}`);\n      }\n"
    "      return await response.json();\n    } catch (err) {\n      if (attempt === retries) throw err;\n"
    "      const delay = baseDelayMs * 2 ** attempt;\n",
    # 5. math word problem
    "A water tank is being filled by two pipes and emptied by a third. Pipe A alone can fill the empty "
    "tank in 6 hours, and pipe B alone can fill it in 4 hours. Pipe C alone can empty a full tank in 12 "
    "hours. The tank starts empty at 8:00 in the morning with all three pipes open. At 10:00 pipe B is "
    "closed for maintenance and reopened at 11:30. We want to know the exact time at which the tank "
    "becomes full.\n\nSolution. First compute each pipe's rate in tanks per hour. Pipe A fills",
    # 6. JSON
    '{\n  "store": "Northside Hardware",\n  "inventory": [\n    {"sku": "HX-1001", "name": "claw hammer", '
    '"price": 18.5, "quantity": 42, "tags": ["tools", "hand"]},\n    {"sku": "HX-1002", "name": "tape '
    'measure 25 ft", "price": 12.0, "quantity": 17, "tags": ["tools", "measuring"]},\n    {"sku": "HX-2040", '
    '"name": "cordless drill", "price": 129.99, "quantity": 6, "tags": ["tools", "power"]},\n    {"sku": ',
    # 7. recipe
    "Classic French onion soup (serves four)\n\nIngredients:\n- 6 large yellow onions, thinly sliced\n"
    "- 4 tablespoons unsalted butter\n- 1 tablespoon olive oil\n- 1 teaspoon sugar\n- 2 cloves garlic, minced\n"
    "- 1/2 cup dry white wine\n- 6 cups beef stock\n- 2 sprigs fresh thyme and 1 bay leaf\n"
    "- 1 baguette, sliced and toasted\n- 2 cups grated Gruyere\n\nMethod:\n1. Melt the butter with the oil "
    "in a heavy pot over medium heat. Add the onions and sugar and stir to coat.\n2.",
    # 8. Chinese prose
    "丝绸之路是古代连接东亚与地中海世界的一系列贸易路线的总称。它的名字来源于中国出口的丝绸，但沿途交换的商品远不止丝绸，"
    "还包括香料、玻璃、金银器、马匹以及纸张。更重要的是，宗教、技术和思想也沿着这些道路传播，例如佛教从印度传入中国，"
    "造纸术则由中国西传。汉代张骞出使西域被视为这条道路正式开通的标志。到了唐代，长安成为世界上最繁华的城市之一，",
    # 9. English with emoji
    "Weekend recap 🌄🥾 We finally did the ridge trail! Started at 5am in the dark with headlamps 🔦, "
    "reached the first lookout just as the sun came up 🌅 and honestly it was worth every blister. "
    "Lunch was squashed sandwiches 🥪 and the best peaches 🍑 of my life at the lake. Saw two marmots, "
    "one very suspicious goat 🐐, and zero bears 🐻 (thank goodness). The descent destroyed my knees 😵‍💫 "
    "but we made it back before the storm ⛈️ rolled in. Next month we want to try",
    # 10. legal text
    "7. Limitation of Liability. To the maximum extent permitted by applicable law, in no event shall "
    "the Provider, its affiliates, or their respective officers, directors, employees, or agents be "
    "liable for any indirect, incidental, special, consequential, or punitive damages, including "
    "without limitation loss of profits, data, use, goodwill, or other intangible losses, resulting "
    "from (a) your access to or use of or inability to access or use the Service; (b) any conduct or "
    "content of any third party on the Service; or (c) unauthorized access, use, or alteration of your",
    # 11. dialogue
    "INT. LIGHTHOUSE - NIGHT\n\nRain hammers the windows. MARGARET (60s), the keeper, trims a wick by "
    "lamplight. A knock at the door. She freezes, then opens it to find TOMAS (20s), soaked and shivering.\n\n"
    "TOMAS\nPlease. My boat went onto the rocks past the point. I didn't know where else to go.\n\n"
    "MARGARET\nNobody comes out here in weather like this. Nobody sensible.\n\nTOMAS\nI never said I was "
    "sensible.\n\nShe studies him for a long moment, then steps aside.\n\nMARGARET\n",
    # 12. poetry
    "When autumn lays its copper on the hill\nand swallows stitch the evening into rows,\nthe orchard "
    "keeps a kind of patient still,\nas if it knows what every orchard knows:\nthat what is gathered "
    "must at last be spent,\nthat sweetness is a debt the branches pay,\nthat every leaf the long green "
    "summer lent\nis called back gently, one by one, away.\nThe crows convene along the fence to talk\n"
    "of frost, and every gate is left ajar;\nAnd I, who walk the furrows after rain\n"
    "with nothing in my hands but what is lost,\n",
    # 13. SQL
    "-- Schema for a small library system\nCREATE TABLE authors (\n  id SERIAL PRIMARY KEY,\n"
    "  name TEXT NOT NULL,\n  born DATE\n);\n\nCREATE TABLE books (\n  id SERIAL PRIMARY KEY,\n"
    "  title TEXT NOT NULL,\n  author_id INTEGER REFERENCES authors(id),\n  published_year INTEGER,\n"
    "  copies_available INTEGER DEFAULT 0\n);\n\nCREATE TABLE loans (\n  id SERIAL PRIMARY KEY,\n"
    "  book_id INTEGER REFERENCES books(id),\n  borrower TEXT NOT NULL,\n  loaned_at TIMESTAMP NOT NULL,\n"
    "  returned_at TIMESTAMP\n);\n\n-- Find every author with more than three books currently on loan\nSELECT",
    # 14. markdown table
    "## Quarterly results\n\n| Region | Q1 revenue | Q2 revenue | Change |\n|---|---|---|---|\n"
    "| North America | $4.2M | $4.9M | +16.7% |\n| Europe | $3.1M | $3.0M | -3.2% |\n"
    "| Asia-Pacific | $2.4M | $3.3M | +37.5% |\n| Latin America | $0.8M | $0.9M | +12.5% |\n\n"
    "### Notes\n\nGrowth in Asia-Pacific was driven mostly by the enterprise tier, where three large "
    "contracts closed in May. The decline in Europe reflects currency effects rather than lower volume; "
    "in constant currency the region grew",
    # 15. news report
    "RIVERTON — City council members voted 5 to 2 on Tuesday night to approve a plan that will convert "
    "two downtown parking garages into mixed-use buildings with apartments, shops, and a public "
    "library branch. The proposal, which has been debated for more than a year, drew a crowd of over "
    "two hundred residents to the council chambers. Supporters argued that the city badly needs "
    "housing near its transit hub, while opponents warned that removing nearly nine hundred parking "
    "spaces would hurt small businesses. Mayor Elena Park, who cast the deciding vote on the committee,",
    # 16. technical docs
    "HTTP caching relies on a handful of response headers. Cache-Control is the primary one: max-age "
    "tells a cache how many seconds a response stays fresh, no-store forbids caching entirely, and "
    "private restricts storage to the end user's browser rather than shared proxies. When a cached "
    "response goes stale, the client does not have to download it again from scratch. Instead it can "
    "send a conditional request using If-None-Match, carrying the ETag it received earlier, or "
    "If-Modified-Since, carrying the Last-Modified date. If the resource has not changed, the server",
    # 17. Spanish
    "La cocina mediterránea se basa en ingredientes sencillos y de temporada: aceite de oliva, verduras "
    "frescas, legumbres, pescado y cereales integrales. Más que una lista de recetas, es una forma de "
    "comer y de compartir la mesa que se ha transmitido de generación en generación. Los estudios han "
    "relacionado este patrón alimentario con un menor riesgo de enfermedades cardiovasculares, aunque "
    "los investigadores insisten en que también influyen la actividad física y la vida social. En "
    "España, por ejemplo, el desayuno suele ser ligero, mientras que la comida principal",
    # 18. Japanese
    "日本の鉄道網は、その正確さと安全性で世界的に知られている。新幹線は一九六四年、東京オリンピックの開幕に合わせて東京と"
    "新大阪の間で開業し、当時としては画期的な時速二百キロを超える営業運転を実現した。それ以来、路線は北海道から九州まで"
    "延び、平均の遅れは一分にも満たないと言われる。こうした運行を支えているのは、",
    # 19. fantasy story
    "The map had been in the family for four generations, and for four generations nobody had been "
    "able to read it. It was drawn on something thinner than paper and tougher than leather, and the "
    "coastline it showed matched no coast that Ilse had ever seen in any atlas. Her grandmother had "
    "kept it folded inside a hymnal. Her mother had tried to sell it twice. Ilse, who had inherited "
    "both the map and the debts, spread it on the kitchen table the night after the funeral and "
    "noticed, for the first time, that the ink moved when",
    # 20. trivia list
    "Twelve surprising facts about the ocean:\n1. More than eighty percent of the ocean has never been "
    "mapped in detail.\n2. The Pacific is wider than the Moon.\n3. Some deep-sea fish produce their own "
    "light using symbiotic bacteria.\n4. The longest mountain range on Earth, the mid-ocean ridge, is "
    "almost entirely underwater.\n5. Sound travels about four times faster in seawater than in air.\n"
    "6. The Mariana Trench is deep enough to swallow Mount Everest with more than two kilometers to spare.\n7.",
]
