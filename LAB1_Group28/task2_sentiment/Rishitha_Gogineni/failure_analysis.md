# Task 2 Error Analysis

The manual review uses the final TextCNN model because it had the highest test accuracy among my three final models. I reviewed 20 incorrect test predictions using the four groups required in the lab. The full selected review text is saved in `outputs/error_analysis/textcnn_manual_error_review.csv`.

## Confident false positives

### Case 1

Test index: 29330

True label: Negative

Predicted label: Positive

Positive probability: 0.9999

Error type: Possible label noise

Review excerpt: Wow love the place and everything is very clean and new!  Great place to come and relax worth a try!  Cheers,  Eric Van Nguyen Visited April 2012

Observation: The review is strongly positive from start to finish even though the stored label is negative.

Testable fix: Audit or relabel ambiguous training examples, or use a loss that is more robust to noisy labels.

### Case 2

Test index: 17407

True label: Negative

Predicted label: Positive

Positive probability: 0.9996

Error type: Long mixed review and truncation

Review excerpt: This was my second time dining at Company. And unfortunately, it will probably be the last. Maybe that's not fair. I'll explain.  I first dined at Company back in June. My party of six was seated by a waiter and looked after very attentively by three support servers clad all black pants, collared shirts and aprons. I felt a little under-dressed. The menu w...

Observation: The review starts with a very positive earlier visit and later turns negative. With a 384-token limit, the model can over-weight the positive section.

Testable fix: Use head-and-tail truncation or sentence-level pooling so the end of long reviews is retained.

### Case 3

Test index: 12495

True label: Negative

Predicted label: Positive

Positive probability: 0.9996

Error type: Mixed sentiment

Review excerpt: Nice place, good decor.  A little slow on the service part.  Sushi makes up for it.  I enjoyed the baked rolls very much. Salmon tasted good. Calamari was surprisingly yummy. J Roll came out cold. (Deep fried roll)  AYCE: $26.99 dinner.  Expensive..  I've had better, but I just like the location. Don't come too often. Kaya all the way.

Observation: Positive food words dominate, while the negative judgment is mainly about price and overall preference.

Testable fix: Add aspect-aware pooling so food quality and value can contribute separately.

### Case 4

Test index: 22663

True label: Negative

Predicted label: Positive

Positive probability: 0.9993

Error type: Aspect conflict

Review excerpt: About average so far as steakhouses go. My rib eye was very tasty but a little over cooked. I didn't complain because the flavor was still fantastic. I thought the process were quite out of order. I've had better for sell. Over all I'd say they are more for show than content. If you want to have a great high end steak I'd recommend Ruth Chris, LG's, Flemings...

Observation: Food, service, and ambience are described positively, but price and value drive the negative label.

Testable fix: Use sentence or aspect-level features instead of one global pooled decision.

### Case 5

Test index: 35584

True label: Negative

Predicted label: Positive

Positive probability: 0.9992

Error type: Mixed pros and cons

Review excerpt: Got a $10 for $20 check in deal with yelp. Me and my girls really needed a massage after walking around Vegas all night in our 6 inch heels. Pros: amazing decor, skilled massages, tea service afterward. Cons: staff is awkward. My sister's therapist answered TWO cell phone calls during her hour service. Therefore she was done about 12 minutes after the rest o...

Observation: The review contains strong positive phrases about the massage along with clear service complaints.

Testable fix: Train with sentence-level sentiment aggregation to handle opposing aspects in one review.

## Confident false negatives

### Case 1

Test index: 29494

True label: Positive

Predicted label: Negative

Positive probability: 0.0000

Error type: Update reversal

Review excerpt: UPDATED.   My initial very frustrated and dramatic review read as follows:  Bililng practices are at best negligent and at worst fraudulent.   I started going to the studio per a groupon. I enjoyed the experience so much that I purchased a discounted 20 pack of classes. When I purchased the classes, my credit card was charged. However, the purchase did...

Observation: A strongly negative original review is followed by a later update describing the owner trying to fix the problem.

Testable fix: Give more weight to explicit update sections or use recency-aware sentence pooling.

### Case 2

Test index: 30366

True label: Positive

Predicted label: Negative

Positive probability: 0.0000

Error type: Mixed overall rating

Review excerpt: We have eaten at the restaurant several times as we enjoy the food and last night brought friends (another couple) as it was one of their birthdays. In the afternoon called and spoke with a hostess to tell her would like \""happy birthday\"" written on the plate on whatever dessert was ordered. When we arrived at the restaurant ran in ahead and mentioned it...

Observation: The food and waiter are praised, but the birthday request and manager response are criticized.

Testable fix: Use aspect-level sentiment features and combine them before the final prediction.

### Case 3

Test index: 30086

True label: Positive

Predicted label: Negative

Positive probability: 0.0001

Error type: Context and label ambiguity

Review excerpt: Went there yesterday and found that the place has just CLOSED. Another victim of a good chef with a bad location and business planning.   Hint for future restauranteurs: Don't open a Scottsdale restaurant in the summer unless you have the cash to survive with no business till the following March, advertise heavily, and don't locate your business where cust...

Observation: The text reports that the business closed and gives business advice, while the chef is still described positively.

Testable fix: Flag low-agreement examples for review and use robust training for ambiguous labels.

### Case 4

Test index: 29600

True label: Positive

Predicted label: Negative

Positive probability: 0.0001

Error type: Local negative phrase dominates

Review excerpt: Jimmy Johns is so fast, and with all that extra time I get back, I like to take my gold over to the \""i buy gold\"" place right next door.  Terrible joke, sorry.  I dunno it's decent food, and they got crushed ice. My favorite order is a turkey tom with mustard. Check it out. :D

Observation: The phrase "Terrible joke, sorry" is negative, but the review then says the food is decent and recommends an order.

Testable fix: Use wider context around sentiment words so isolated negative phrases do not dominate.

### Case 5

Test index: 22807

True label: Positive

Predicted label: Negative

Positive probability: 0.0001

Error type: Edited review conflict

Review excerpt: EDIT: They really did change the service up since I last posted this.  Horrible service.  Used to be my favorite pizza in the city (at a reasonable price), but I'm rethinking that. We just had an altercation with a server who refused to split a check when we were paying with cash. He then proceeded to disrespect the party at the table, telling us to 'not...

Observation: The opening edit says service improved, but most of the remaining text describes the older negative experience.

Testable fix: Detect edited or updated sections and weight the newest statement more heavily.

## Near-threshold errors

### Case 1

Test index: 15220

True label: Positive

Predicted label: Negative

Positive probability: 0.4999

Error type: Mixed sentiment near threshold

Review excerpt: Tryna have a good time ASAP?! 44 Magnum + 2 extra shots. You'll definitely be on a good level while enjoying Vegas.  *warning if yo dont wanna experience flavored vodka i guess sticking with one extra shot will still give you some level.  Eh, felt like customer service was whatever. seemed a bit rude serving but its understandable to those who wanna jus...

Observation: Positive comments about having fun are offset by a complaint about rude service, leaving the model almost exactly at 0.5.

Testable fix: Use sentence-level scores to separate experience quality from service quality.

### Case 2

Test index: 25008

True label: Negative

Predicted label: Positive

Positive probability: 0.5001

Error type: Factual negative review near threshold

Review excerpt: CAUTION---In spite of the similar name, they are NOT the American Cancer Society. They called me tonight seeking a donation, and when I interrupted the fast-talking womans script to ask how much money goes to fund-raising expenses, she answered it was an 85%/15% split.  She paused right there, and asked if \""These special women can count on me\""; so I as...

Observation: The review is mostly factual and numerical, with relatively few common sentiment words.

Testable fix: Add character or phrase features that capture cautionary language and negative framing.

### Case 3

Test index: 1161

True label: Negative

Predicted label: Positive

Positive probability: 0.5002

Error type: Sarcasm

Review excerpt: Be sure to ask if your car is ready yet. They forget to announce it sometimes and leave you in the lobby waiting for Christmas.

Observation: The negative point is expressed indirectly through the joke about waiting for Christmas.

Testable fix: Add training examples with sarcasm or use a context model that can learn indirect sentiment cues.

### Case 4

Test index: 5144

True label: Positive

Predicted label: Negative

Positive probability: 0.4998

Error type: Negation and reference

Review excerpt: I started picking up books here for class, and began to really appreciate their other selections. Certainly not a place to look for Sarah Palin's autobiography - plus one star for that. They've also got a cool rack of zines and a bunch of bumper stickers as well.

Observation: The sentence about Sarah Palin is a joke using negation, while the rest of the bookstore review is positive.

Testable fix: Use stronger negation handling and contextual phrase features rather than isolated word cues.

### Case 5

Test index: 5443

True label: Negative

Predicted label: Positive

Positive probability: 0.5010

Error type: Weak sentiment or label noise

Review excerpt: Really cool lights at the fair tonight look to the north and see such a delight. Well last night was the last day of the fair is closed it's gone now for next year but they will be the next fair in the spring.

Observation: The text is mostly neutral to positive and does not contain a clear negative opinion despite the negative label.

Testable fix: Audit weakly expressed labels and use confidence-aware training for uncertain examples.

## Slice-specific failures

### Case 1

Test index: 25750

True label: Negative

Predicted label: Positive

Positive probability: 0.5218

Error type: Long review and tail truncation

Review excerpt: I've been a customer here for a number of years. Always had good service until recently. I went into the store to have batteries and service done on five watches. While I was there I noticed they started carrying Glycine watches which I've had my eye on for a while.  A young salesperson quoted me a price of $2,900.00 for one of the Airmen Multi-Time Zone w...

Observation: A long history of good service is followed by a pricing dispute. Important final context can be lost after 384 tokens.

Testable fix: Use head-and-tail truncation or hierarchical sentence pooling for long reviews.

### Case 2

Test index: 25266

True label: Negative

Predicted label: Positive

Positive probability: 0.9951

Error type: Long mixed narrative

Review excerpt: HUGE COCKROACH IN BATHROOM OF OUR ROOM!!!!  Let me begin by saying how excited my husband and I were for this trip, and especially to stay at The Cosmopolitan - the newest, hottest hotel on the strip. We've stayed at 5 different other 5 star hotels so far in previous visits, but were sure this one looked even better. We've had this trip planned for over a y...

Observation: The review mixes excited positive hotel descriptions with a severe complaint. Many positive local phrases can overwhelm the overall negative outcome.

Testable fix: Aggregate sentiment across sentences and give stronger weight to explicit complaint sentences.

### Case 3

Test index: 9911

True label: Positive

Predicted label: Negative

Positive probability: 0.4259

Error type: Long mixed experience

Review excerpt: Gather around little Yelpers, let me tell you about my stay at ye ol' Treasure Island. No, but seriously get ready to hear it.  We had the penthouse suite here. Our room was beautifully decorated and looked real crisp. Being an all guys birthday weeknd, we wanted to go all out. As soon as entering the room, we immediately fired up the jacuzzi. And 10 minut...

Observation: The review contains many positive hotel details along with concerns about staff monitoring and other problems.

Testable fix: Use hierarchical pooling so the model can combine sentence-level evidence across the whole review.

### Case 4

Test index: 4664

True label: Negative

Predicted label: Positive

Positive probability: 0.9290

Error type: Long review and temporal contrast

Review excerpt: I've been here twice. It was like visiting two completely different restaurants.  The first time my girlfriend and I tried this place on a Friday night about a month ago, we loved it. It's a small place, clean and inviting, with a neighborhood-hangout kind of vibe. We'd heard the wings were good, and we weren't disappointed; the garlic Parmesan wings were...

Observation: The first visit is described very positively and the second visit is negative. The early positive section can dominate, especially with truncation.

Testable fix: Keep both the beginning and end of long reviews and model temporal contrast between visits.

### Case 5

Test index: 37318

True label: Negative

Predicted label: Positive

Positive probability: 0.9851

Error type: Long setup before criticism

Review excerpt: The Yusho stands for   YU SHOuld not go to this place.  At first i saw the sign and i was super excited I could not wait for this place to open it's doors because i love Japanese style grilling. Yakitori and Robatayaki is the best and they have noodles too? Bonus!!!!  Every time i drove by the sign \""Yusho Japanese Grill & Noodle House\"" I could just...

Observation: The review spends many tokens describing positive expectations before explaining why the experience was poor.

Testable fix: Use head-and-tail input or sentence selection so the final judgment is always visible to the classifier.
