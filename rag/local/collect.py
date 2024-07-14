import json
import os

# The JSON data
data = {
  "BABY-001": "Convertible Crib by TinyDreams. Color: Soft White. Material: Solid pine wood with non-toxic finish. Dimensions: 54 inches (length), 30 inches (width), 40 inches (height). Features adjustable mattress height, converts to toddler bed and daybed. Includes safety guardrails and under-crib storage drawer.",
  "BABY-002": "Multi-Function Diaper Bag by ParentEase. Color: Charcoal Gray. Material: Water-resistant polyester with insulated pockets. Dimensions: 16 inches (height), 14 inches (width), 8 inches (depth). Features multiple compartments, changing pad, and stroller straps. Ergonomically designed with padded shoulder straps.",
  "BABY-003": "Digital Baby Monitor by SafeWatch. Color: Pearl White. Range: 1000 feet. Features 5-inch LCD screen, night vision, temperature sensor, and two-way audio. Includes secure FHSS technology for interference-free connection. Rechargeable battery with 12-hour runtime and wall-mountable camera.",
  "CAR-001": "Universal Car Phone Mount by DriveTech. Color: Black. Material: ABS plastic with silicone grips. Dimensions: Adjustable to fit phones 4.7-6.5 inches. Features 360-degree rotation, one-touch release button, and air vent clip. Suitable for all car models, easy to install and remove.",
  "CAR-002": "Leather Steering Wheel Cover by LuxAuto. Color: Jet Black. Material: Genuine leather with microfiber lining. Dimensions: Fits steering wheels 14.5-15.5 inches in diameter. Features anti-slip design, breathable texture, and easy installation. Enhances grip and adds a touch of luxury to the interior.",
  "CAR-003": "Portable Jump Starter by PowerBoost. Color: Red/Black. Capacity: 12000mAh. Features 800A peak current, built-in LED flashlight, and USB ports for device charging. Suitable for 12V vehicles (up to 6.0L gas or 4.0L diesel engines). Includes jumper cables, carrying case, and safety protection features.",
  "MUSIC-001": "Acoustic Guitar by MelodyMaker. Color: Natural Spruce. Material: Solid spruce top with mahogany back and sides. Dimensions: 41 inches (full size). Features rosewood fretboard, chrome die-cast tuners, and high-gloss finish. Ideal for beginners and intermediate players, delivers rich, warm sound.",
  "MUSIC-002": "Digital Keyboard by SoundWave. Color: Black. Keys: 88 weighted keys. Features 600 tones, 195 rhythms, and built-in speakers. Dimensions: 52 inches (length), 11 inches (width), 5 inches (height). Includes sustain pedal, music stand, and power adapter. Perfect for practice and performance.",
  "MUSIC-003": "Electronic Drum Set by RhythmPro. Color: Matte Black. Features 8 drum pads, 3 cymbals, and adjustable rack. Dimensions: 48 inches (width), 24 inches (depth), 30 inches (height). Includes drum module with 300+ sounds, metronome, and recording capability. Suitable for home practice and live gigs.",
  "CHAIR-001": "Ergonomic Office Chair by ComfortSeating. Color: Midnight Black. Material: Breathable mesh with memory foam cushioning. Adjustable lumbar support, headrest, and armrests. Dimensions: 45-53 inches (height adjustable), 24 inches (width), 24 inches (depth). Perfect for long hours of sitting with 360-degree swivel and smooth-rolling casters.",
  "CHAIR-002": "Classic Wooden Dining Chair by OakHaven. Color: Warm Walnut. Material: Solid oak wood with a high-gloss finish. Cushioned seat with beige fabric upholstery. Dimensions: 35 inches (height), 18 inches (width), 20 inches (depth). Traditional design with carved backrest and sturdy build.",
  "CHAIR-003": "Modern Accent Chair by UrbanHome. Color: Slate Gray. Material: Velvet fabric with gold-tone metal legs. Dimensions: 30 inches (height), 28 inches (width), 26 inches (depth). Sleek design suitable for living rooms or bedrooms. Features a deep seat and slightly reclined backrest for comfort.",
  "TABLE-001": "Expandable Dining Table by FlexiFurnish. Color: Espresso Brown. Material: Engineered wood with a veneer finish. Dimensions: 60-80 inches (expandable length), 36 inches (width), 30 inches (height). Comes with a hidden butterfly leaf for easy extension. Seats 6-8 people comfortably.",
  "TABLE-002": "Minimalist Coffee Table by NordicDesign. Color: Matte White. Material: MDF with a lacquered finish. Dimensions: 42 inches (length), 24 inches (width), 18 inches (height). Features a single storage drawer and an open shelf below. Perfect for contemporary living rooms.",
  "TABLE-003": "Rustic Farmhouse Table by Countryside Creations. Color: Natural Pine. Material: Reclaimed pine wood with a distressed finish. Dimensions: 72 inches (length), 38 inches (width), 30 inches (height). Features a sturdy trestle base and a planked top for a rustic look. Seats 6-8 people.",
  "SOFA-001": "L-Shaped Sectional Sofa by CozyCorner. Color: Charcoal Gray. Material: Soft polyester blend fabric with high-density foam cushions. Dimensions: 98 inches (length), 64 inches (chaise length), 35 inches (height). Features reversible chaise and removable cushion covers. Perfect for large living spaces.",
  "SOFA-002": "Convertible Sleeper Sofa by MultiComfort. Color: Navy Blue. Material: Microfiber fabric with innerspring mattress. Dimensions: 82 inches (length), 37 inches (depth), 36 inches (height). Converts easily into a queen-size bed. Ideal for small apartments or guest rooms.",
  "SOFA-003": "Chesterfield Sofa by HeritageLux. Color: Emerald Green. Material: Velvet fabric with deep button tufting. Dimensions: 85 inches (length), 35 inches (depth), 32 inches (height). Features rolled arms and a solid wood frame. Adds a touch of elegance to any living room."
}

# Directory to save individual JSON files
output_dir = "./data"
os.makedirs(output_dir, exist_ok=True)

# Split JSON data and save each SKU and description in a separate file
for sku, description in data.items():
    individual_data = description
    file_path = os.path.join(output_dir, f"{sku}.txt")
    with open(file_path, 'w') as file:
        file.write(individual_data)

print(f"All individual JSON files have been saved in the '{output_dir}' directory.")
