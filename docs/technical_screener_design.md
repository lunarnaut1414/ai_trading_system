# Technical Screener Design Documentation

## Architecture Overview

The Technical Screener is a sophisticated pattern recognition system that analyzes market data to identify high-probability trading opportunities through technical analysis. It consists of four main components working in concert to deliver actionable trading signals.

## Core Components

### 1. Pattern Recognition Engine (`pattern_recognition.py`)

The pattern recognition engine implements detection algorithms for 17+ distinct chart patterns, organized into six categories:

**Triangular Patterns**
- Ascending Triangle: Horizontal resistance with rising support
- Descending Triangle: Horizontal support with falling resistance  
- Symmetrical Triangle: Converging trendlines with prior trend continuation bias

**Momentum Patterns**
- Bull Flag: Strong upward move followed by consolidation
- Bear Flag: Strong downward move followed by consolidation
- Pennant: Similar to flags but with converging trendlines

**Breakout Patterns**
- Resistance Breakout: Price exceeds established resistance with volume
- Support Breakout: Price falls below established support
- Volume Breakout: Unusual volume spike indicating potential move

**Reversal Patterns**
- Double Top: Two peaks at similar levels indicating bearish reversal
- Double Bottom: Two troughs at similar levels indicating bullish reversal
- Head & Shoulders: Three-peak reversal pattern
- Inverse Head & Shoulders: Three-trough reversal pattern

**Trend Patterns**
- Uptrend Channel: Parallel rising support and resistance
- Downtrend Channel: Parallel falling support and resistance
- Sideways Channel: Horizontal consolidation range

**Complex Patterns**
- Cup and Handle: U-shaped accumulation with breakout
- Rising Wedge: Converging upward lines (bearish)
- Falling Wedge: Converging downward lines (bullish)

### 2. Technical Screener (`technical_screener.py`)

The main screener orchestrates the scanning process:

**Universe Management**
- Maintains S&P 500 and NASDAQ constituent lists
- Applies liquidity filters (minimum 500K daily volume)
- Market cap filtering ($1B minimum)
- Dynamic universe updates

**Parallel Processing**
- Concurrent pattern detection across multiple symbols
- ThreadPoolExecutor for CPU-bound operations
- Asynchronous I/O for data fetching
- Batch processing optimization

**Signal Quality Assessment**
- Multi-factor scoring algorithm
- Confidence weighting (0-10 scale)
- Risk/reward ratio calculation
- Pattern completion percentage
- Volume confirmation requirements

**Caching System**
- 4-hour cache duration for screening results
- Symbol-level caching for technical indicators
- Reduces API calls and processing overhead

### 3. Integration Layer (`screener_integration.py`)

Provides formatted outputs for different consumers:

**Junior Analyst Integration**
- Formats technical signals with entry/exit points
- Adds fundamental validation flags
- Provides risk metrics and quality scores
- Includes technical context for analysis

**Market Breadth Analysis**
- Aggregates bullish/bearish signal counts
- Calculates market sentiment indicators
- Pattern distribution analysis
- Quality metrics across the market

**Portfolio Monitoring**
- Screens existing holdings for technical signals
- Generates action recommendations
- Identifies positions requiring attention
- Risk alert generation

### 4. Scheduling System (`screening_scheduler.py`)

Automates daily screening operations:

**Scheduled Scans**
- Post-market comprehensive scan (4:30 PM ET)
- Pre-market priority scan (8:00 AM ET)
- Weekend deep analysis runs
- Intraday alert monitoring

## Data Flow Architecture

```
Market Data (AlpacaProvider)
    ↓
Technical Screener
    ├── Universe Filter (liquidity/market cap)
    ├── Pattern Recognition Engine
    │   ├── Price Data Analysis
    │   ├── Volume Analysis
    │   └── Indicator Calculation
    ├── Signal Generation
    │   ├── Pattern Detection
    │   ├── Confidence Scoring
    │   └── Risk/Reward Analysis
    └── Output Formatting
        ├── Junior Analyst Format
        ├── Portfolio Manager Format
        └── Database Storage
```

## Signal Quality Scoring Algorithm

The quality score for each signal is calculated using:

```python
quality_score = (
    confidence * 0.4 +                    # Pattern confidence weight
    min(risk_reward_ratio, 5) * 2 +      # Risk/reward capped at 5
    (10 if volume_confirmed else 5) * 0.2 + # Volume confirmation
    pattern_completion / 10 * 0.2        # Pattern completion percentage
)
```

## Test Suite Design

### Unit Test Coverage (20 Tests)

#### Pattern Recognition Tests (6 tests)

1. **Engine Initialization Test**
   - Verifies default parameters
   - Checks threshold settings
   - Validates configuration

2. **Support/Resistance Detection Test**
   - Tests level identification algorithm
   - Validates touch counting logic
   - Checks strength calculation

3. **Triangle Pattern Detection Test**
   - Tests ascending/descending/symmetrical triangles
   - Validates trendline detection
   - Checks breakout level calculation

4. **Breakout Pattern Detection Test**
   - Tests resistance/support breakouts
   - Validates volume confirmation
   - Checks signal generation

5. **Indicator Calculation Test**
   - Tests RSI, MACD, Bollinger Bands
   - Validates moving average calculations
   - Checks ATR and volatility metrics

6. **Signal Filtering Test**
   - Tests quality threshold filtering
   - Validates confidence requirements
   - Checks risk/reward filtering

#### Technical Screener Tests (7 tests)

7. **Screener Initialization Test**
   - Verifies component setup
   - Checks default parameters
   - Validates database connection

8. **Universe Initialization Test**
   - Tests symbol loading
   - Validates liquidity filtering
   - Checks universe combination

9. **Single Symbol Analysis Test**
   - Tests pattern detection pipeline
   - Validates data transformation
   - Checks signal generation

10. **Signal Ranking Test**
    - Tests quality score calculation
    - Validates sorting algorithm
    - Checks per-ticker limits

11. **Opportunity Formatting Test**
    - Tests output structure
    - Validates priority assignment
    - Checks data completeness

12. **Cache Functionality Test**
    - Tests cache storage/retrieval
    - Validates TTL enforcement
    - Checks cache invalidation

13. **Error Handling Test**
    - Tests invalid symbol handling
    - Validates empty data handling
    - Checks exception recovery

#### Scheduler Tests (3 tests)

14. **Scheduler Initialization Test**
    - Verifies schedule setup
    - Checks time configuration
    - Validates thread management

15. **Scheduler Start/Stop Test**
    - Tests scheduler lifecycle
    - Validates thread cleanup
    - Checks state management

16. **Callback Registration Test**
    - Tests event handler registration
    - Validates callback invocation
    - Checks error handling

#### Integration Tests (4 tests)

17. **Junior Analyst Opportunities Test**
    - Tests opportunity formatting
    - Validates filtering logic
    - Checks data structure

18. **Market Breadth Analysis Test**
    - Tests sentiment calculation
    - Validates aggregation logic
    - Checks metric calculation

19. **Portfolio Screening Test**
    - Tests holdings analysis
    - Validates signal grouping
    - Checks recommendation generation

20. **Complete Workflow Test**
    - End-to-end integration test
    - Tests full screening pipeline
    - Validates all components

### Test Data Strategy

**Mock Data Generation**
- Realistic OHLCV data with controlled patterns
- Configurable volume levels for liquidity testing
- Time-series data with proper timestamps
- Edge cases for pattern detection

**Fixture Design**
- Reusable mock providers
- Sample price data generators
- Pre-configured test signals
- Integration test scaffolding

### Test Metrics

**Coverage Goals**
- Line coverage: >80%
- Branch coverage: >70%
- Integration coverage: 100% of API endpoints

**Performance Benchmarks**
- Single symbol analysis: <100ms
- 100-symbol screen: <10 seconds
- Cache retrieval: <10ms

## Performance Optimization

### Concurrency Strategy
- ThreadPoolExecutor for CPU-bound pattern detection
- Async I/O for network operations
- Batch processing for multiple symbols
- Connection pooling for database operations

### Caching Strategy
- L1: In-memory cache for hot data
- L2: Database cache for historical patterns
- TTL-based invalidation
- Lazy loading for cold data

### Resource Management
- Maximum 10 concurrent API requests
- Thread pool size: 5 workers
- Memory limit: 2GB for pattern detection
- Database connection pool: 10 connections

## Integration Points

### Input Sources
- AlpacaDataProvider: Real-time and historical market data
- Database: Historical patterns and signals
- Configuration: Screening parameters and thresholds

### Output Consumers
- Junior Research Analyst: Fundamental validation
- Portfolio Manager: Position decisions
- Analytics Agent: Performance tracking
- Database: Signal persistence

## Error Handling

### Graceful Degradation
- Falls back to cached data on API failure
- Uses default universe on initialization failure
- Continues with partial results on symbol errors
- Logs all errors for debugging

### Recovery Mechanisms
- Automatic retry with exponential backoff
- Circuit breaker for repeated failures
- Alternative data source fallback
- Manual override capabilities

## Future Enhancements

### Planned Features
- Machine learning pattern validation
- Real-time streaming pattern detection
- Custom pattern definition framework
- Backtesting integration
- Multi-timeframe analysis

### Scalability Improvements
- Distributed processing across multiple nodes
- GPU acceleration for pattern detection
- Event-driven architecture
- Microservices decomposition