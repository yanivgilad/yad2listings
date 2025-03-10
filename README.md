# Yad2 Vehicle Price Analyzer

Tool for scraping and visualizing vehicle pricing data from Yad2.

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/yad2-vehicle-analyzer.git
cd yad2-vehicle-analyzer

# Install dependencies
pip install -r requirements.txt
```

## Examples

Basic usage:
```bash
# Run with default settings (Toyota bZ4X)
python vehicle_analyzer.py
```

Scrape specific vehicle model:
```bash
# Volkswagen ID.4
python vehicle_analyzer.py --manufacturer 41 --model 11579

# Hyundai Ioniq 5
python vehicle_analyzer.py --manufacturer 21 --model 11239

# Nissan Qashqai
python vehicle_analyzer.py --manufacturer 32 --model 10449
```

Use existing data:
```bash
# Skip scraping
python vehicle_analyzer.py --skip-scrape
```

Change web server port:
```bash
python vehicle_analyzer.py --port 8080
```

Find manufacturer and model IDs in Yad2 URLs:
```
https://www.yad2.co.il/vehicles/cars?manufacturer=19&model=12894
```