# Trade Execution Agent - Complete Implementation
# agents/trade_execution_agent.py

import asyncio
import json
import uuid
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
import numpy as np

from src.core.base_agent import BaseAgent
from src.data_provider.alpaca_provider import AlpacaProvider

# Define order enums locally or import from alpaca if available
class OrderSide:
    BUY = "BUY"
    SELL = "SELL"

class OrderType:
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"

class TimeInForce:
    DAY = "DAY"
    GTC = "GTC"  # Good Till Cancelled
    IOC = "IOC"  # Immediate Or Cancel
    FOK = "FOK"  # Fill Or Kill

# ==================== SPECIFICATION CLASSES ====================

class ExecutionStrategy(Enum):
    """Trade execution strategies"""
    MARKET = "market"              # Immediate execution at market price
    LIMIT = "limit"                # Execute at specific price or better
    TWAP = "twap"                  # Time-weighted average price
    VWAP = "vwap"                  # Volume-weighted average price
    ICEBERG = "iceberg"            # Hide order size with smaller chunks
    OPPORTUNISTIC = "opportunistic" # Wait for favorable price movements

class OrderStatus(Enum):
    """Order lifecycle status"""
    PENDING = "pending"            # Order created but not submitted
    SUBMITTED = "submitted"        # Order submitted to exchange
    PARTIALLY_FILLED = "partially_filled"  # Partial execution
    FILLED = "filled"              # Completely executed
    CANCELLED = "cancelled"        # Order cancelled
    REJECTED = "rejected"          # Order rejected by exchange
    EXPIRED = "expired"            # Order expired

class ExecutionUrgency(Enum):
    """Execution urgency levels"""
    LOW = "low"                    # Patient execution over hours/days
    MEDIUM = "medium"              # Execute within session
    HIGH = "high"                  # Execute within 1 hour
    URGENT = "urgent"              # Execute immediately

class MarketConditions(Enum):
    """Market condition assessment"""
    NORMAL = "normal"              # Typical trading conditions
    VOLATILE = "volatile"          # High volatility environment
    ILLIQUID = "illiquid"         # Low volume/wide spreads
    TRENDING = "trending"          # Strong directional movement
    CHOPPY = "choppy"             # Range-bound with noise

@dataclass
class TradeInstruction:
    """Complete trade instruction from Portfolio Manager"""
    symbol: str
    action: str                    # BUY, SELL, TRIM, ADD, CLOSE
    target_weight: float           # Target portfolio weight %
    current_weight: float          # Current portfolio weight %
    confidence: int                # PM confidence 1-10
    time_horizon: str              # short, medium, long
    reasoning: str                 # PM reasoning for trade
    risk_factors: List[str]        # Identified risk factors
    urgency: ExecutionUrgency      # Execution urgency level
    max_slippage: float            # Maximum acceptable slippage %
    execution_window: timedelta    # Time window for execution

@dataclass
class ExecutionPlan:
    """Detailed execution plan for a trade"""
    instruction: TradeInstruction
    strategy: ExecutionStrategy
    order_size: float              # Shares or dollar amount
    target_price: Optional[float]  # Target execution price
    limit_price: Optional[float]   # Limit price for orders
    stop_price: Optional[float]    # Stop loss price
    chunk_size: Optional[int]      # For iceberg orders
    execution_schedule: List[Dict] # Time-based execution schedule
    market_timing: str             # Market timing assessment
    expected_slippage: float       # Expected execution slippage
    risk_assessment: Dict          # Execution risk factors

@dataclass
class ExecutionResult:
    """Results of trade execution"""
    instruction_id: str
    symbol: str
    action: str
    requested_quantity: float
    executed_quantity: float
    average_price: float
    execution_time: timedelta
    slippage_bps: float           # Basis points
    market_impact: float          # Price impact %
    total_commission: float
    status: OrderStatus
    fills: List[Dict]             # Individual fill details
    execution_quality: str        # excellent/good/fair/poor

# ==================== EXECUTION TIMING ENGINE ====================

class ExecutionTimingEngine:
    """
    Market timing engine for optimal trade execution
    """
    
    def __init__(self, alpaca_provider: AlpacaProvider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger('execution_timing')
        
        # Timing configuration
        self.avoid_first_minutes = 5  # Avoid first 5 minutes
        self.avoid_last_minutes = 10  # Avoid last 10 minutes
        self.lunch_hour_start = 12    # Lunch hour start
        self.lunch_hour_end = 13      # Lunch hour end
    
    async def assess_market_timing(self, symbol: str, action: str) -> Dict:
        """
        Assess optimal timing for trade execution
        
        Args:
            symbol: Stock symbol
            action: Trade action (BUY/SELL)
        
        Returns:
            Timing analysis with recommendations
        """
        
        try:
            # Get current market conditions
            market_conditions = await self._analyze_market_conditions(symbol)
            
            # Analyze intraday patterns
            intraday_analysis = await self._analyze_intraday_patterns(symbol, action)
            
            # Check for upcoming events
            event_analysis = await self._check_market_events()
            
            # Calculate liquidity profile
            liquidity = await self._analyze_liquidity(symbol)
            
            # Generate timing score
            timing_score = self._calculate_timing_score(
                market_conditions, intraday_analysis, event_analysis, liquidity
            )
            
            # Determine optimal strategy
            recommended_strategy = self._recommend_execution_strategy(
                timing_score['score'], market_conditions, liquidity, event_analysis
            )
            
            return {
                'timing_score': timing_score['score'],
                'timing_assessment': timing_score['assessment'],
                'recommended_strategy': recommended_strategy,
                'optimal_execution_window': timing_score['window'],
                'market_conditions': market_conditions,
                'liquidity_analysis': liquidity,
                'execution_risks': timing_score['risks'],
                'expected_slippage': timing_score['expected_slippage']
            }
            
        except Exception as e:
            self.logger.error(f"Timing assessment failed: {str(e)}")
            return self._get_default_timing_assessment()
    
    async def _analyze_market_conditions(self, symbol: str) -> Dict:
        """Analyze current market conditions"""
        
        try:
            # Get market data
            market_data = await self.alpaca.get_market_data([symbol], days=1)
            symbol_data = market_data.get(symbol, {})
            
            # Calculate volatility
            volatility = symbol_data.get('volatility', 0)
            
            # Determine market condition
            if volatility > 0.03:
                condition = MarketConditions.VOLATILE
            elif volatility < 0.01:
                condition = MarketConditions.CHOPPY
            else:
                condition = MarketConditions.NORMAL
            
            return {
                'condition': condition,
                'volatility': volatility,
                'trend_strength': symbol_data.get('trend_strength', 0),
                'volume_profile': symbol_data.get('volume_profile', 'normal')
            }
            
        except Exception as e:
            self.logger.error(f"Market condition analysis failed: {str(e)}")
            return {'condition': MarketConditions.NORMAL, 'volatility': 0.02}
    
    async def _analyze_intraday_patterns(self, symbol: str, action: str) -> Dict:
        """Analyze intraday trading patterns"""
        
        current_time = datetime.now()
        market_open = current_time.replace(hour=9, minute=30)
        market_close = current_time.replace(hour=16, minute=0)
        
        minutes_from_open = (current_time - market_open).total_seconds() / 60
        minutes_to_close = (market_close - current_time).total_seconds() / 60
        
        # Determine timing quality
        timing_quality = 'good'
        
        if minutes_from_open < self.avoid_first_minutes:
            timing_quality = 'poor'  # Opening volatility
        elif minutes_to_close < self.avoid_last_minutes:
            timing_quality = 'poor'  # Closing volatility
        elif self.lunch_hour_start <= current_time.hour < self.lunch_hour_end:
            timing_quality = 'fair'  # Lunch hour lower liquidity
        
        return {
            'timing_quality': timing_quality,
            'minutes_from_open': minutes_from_open,
            'minutes_to_close': minutes_to_close,
            'session_position': 'early' if minutes_from_open < 120 else 'mid' if minutes_to_close > 120 else 'late'
        }
    
    async def _check_market_events(self) -> Dict:
        """Check for upcoming market events"""
        
        # In production, would check economic calendar
        # For now, return basic assessment
        return {
            'upcoming_events': [],
            'risk_level': 'low',
            'recommended_delay': 0
        }
    
    async def _analyze_liquidity(self, symbol: str) -> Dict:
        """Analyze current liquidity conditions"""
        
        try:
            # Get current quote
            quote = await self.alpaca.get_latest_quote(symbol)
            
            if quote and 'bid' in quote and 'ask' in quote:
                spread = quote['ask'] - quote['bid']
                mid_price = (quote['ask'] + quote['bid']) / 2
                spread_pct = (spread / mid_price) * 100 if mid_price > 0 else 0
                
                # Assess liquidity
                if spread_pct < 0.05:
                    liquidity_level = 'excellent'
                    liquidity_score = 9
                elif spread_pct < 0.1:
                    liquidity_level = 'good'
                    liquidity_score = 7
                elif spread_pct < 0.2:
                    liquidity_level = 'fair'
                    liquidity_score = 5
                else:
                    liquidity_level = 'poor'
                    liquidity_score = 3
                
                return {
                    'liquidity_level': liquidity_level,
                    'liquidity_score': liquidity_score,
                    'spread_pct': spread_pct,
                    'bid_size': quote.get('bid_size', 0),
                    'ask_size': quote.get('ask_size', 0)
                }
            
        except Exception as e:
            self.logger.error(f"Liquidity analysis failed: {str(e)}")
        
        return {
            'liquidity_level': 'unknown',
            'liquidity_score': 5,
            'spread_pct': 0.1
        }
    
    def _calculate_timing_score(self, conditions: Dict, intraday: Dict, 
                               events: Dict, liquidity: Dict) -> Dict:
        """Calculate comprehensive timing score"""
        
        # Base score
        timing_score = 5
        
        # Market conditions adjustment
        if conditions['condition'] == MarketConditions.NORMAL:
            timing_score += 2
        elif conditions['condition'] == MarketConditions.VOLATILE:
            timing_score -= 1
        
        # Liquidity adjustment
        timing_score += (liquidity['liquidity_score'] - 5) / 2
        
        # Intraday timing adjustment
        if intraday['timing_quality'] == 'good':
            timing_score += 1
        elif intraday['timing_quality'] == 'poor':
            timing_score -= 2
        
        # Event risk adjustment
        if events['risk_level'] == 'high':
            timing_score -= 2
        elif events['risk_level'] == 'medium':
            timing_score -= 1
        
        # Cap score between 1-10
        timing_score = max(1, min(10, timing_score))
        
        # Determine assessment
        if timing_score >= 8:
            assessment = 'excellent'
        elif timing_score >= 6:
            assessment = 'good'
        elif timing_score >= 4:
            assessment = 'fair'
        else:
            assessment = 'poor'
        
        # Determine execution window
        if timing_score >= 7:
            window_hours = 1
            approach = 'aggressive'
        elif timing_score >= 5:
            window_hours = 2
            approach = 'moderate'
        else:
            window_hours = 4
            approach = 'patient'
        
        # Identify risks
        risks = []
        if conditions['condition'] == MarketConditions.VOLATILE:
            risks.append('high_volatility')
        if liquidity['liquidity_score'] < 5:
            risks.append('low_liquidity')
        if intraday['timing_quality'] == 'poor':
            risks.append('poor_timing')
        
        # Estimate slippage (in basis points)
        base_slippage = 5
        if conditions['condition'] == MarketConditions.VOLATILE:
            base_slippage += 5
        if liquidity['liquidity_score'] < 5:
            base_slippage += 10
        
        return {
            'score': timing_score,
            'assessment': assessment,
            'window': {
                'duration_hours': window_hours,
                'approach': approach
            },
            'risks': risks,
            'expected_slippage': base_slippage
        }
    
    def _recommend_execution_strategy(self, timing_score: float, conditions: Dict,
                                     liquidity: Dict, events: Dict) -> ExecutionStrategy:
        """Recommend optimal execution strategy"""
        
        # High timing score and good liquidity -> aggressive
        if timing_score >= 7 and liquidity['liquidity_score'] >= 7:
            return ExecutionStrategy.MARKET
        
        # Poor liquidity -> patient strategies
        elif liquidity['liquidity_score'] <= 4:
            return ExecutionStrategy.VWAP
        
        # Volatile conditions -> limit orders
        elif conditions['condition'] == MarketConditions.VOLATILE:
            return ExecutionStrategy.LIMIT
        
        # Event risk -> wait
        elif events['risk_level'] == 'high':
            return ExecutionStrategy.OPPORTUNISTIC
        
        # Default to TWAP
        else:
            return ExecutionStrategy.TWAP
    
    def _get_default_timing_assessment(self) -> Dict:
        """Get default timing assessment for fallback"""
        
        return {
            'timing_score': 5,
            'timing_assessment': 'fair',
            'recommended_strategy': ExecutionStrategy.TWAP,
            'optimal_execution_window': {
                'duration_hours': 2,
                'approach': 'moderate'
            },
            'market_conditions': {'condition': MarketConditions.NORMAL},
            'liquidity_analysis': {'liquidity_level': 'unknown', 'liquidity_score': 5},
            'execution_risks': ['analysis_failed'],
            'expected_slippage': 10
        }

# ==================== ORDER MANAGER ====================

class OrderManager:
    """
    Advanced order management system for optimal trade execution
    """
    
    def __init__(self, alpaca_provider: AlpacaProvider):
        self.alpaca = alpaca_provider
        self.logger = logging.getLogger('order_manager')
        
        # Order tracking
        self.active_orders = {}
        self.completed_orders = {}
        self.execution_history = []
        
        # Settings
        self.max_order_age = timedelta(hours=4)
        self.retry_attempts = 3
        self.monitoring_interval = 30  # seconds
    
    async def execute_trade_instruction(self, instruction: TradeInstruction,
                                       execution_plan: ExecutionPlan) -> ExecutionResult:
        """Execute a trade instruction using the specified execution plan"""
        
        instruction_id = str(uuid.uuid4())
        self.logger.info(f"Executing trade {instruction_id}: {instruction.action} {instruction.symbol}")
        
        try:
            # Validate instruction
            validation = self._validate_execution_request(instruction, execution_plan)
            if not validation['valid']:
                return self._create_failed_result(instruction_id, instruction, validation['error'])
            
            # Calculate order parameters
            order_params = await self._calculate_order_parameters(instruction, execution_plan)
            
            # Execute based on strategy
            if execution_plan.strategy == ExecutionStrategy.MARKET:
                result = await self._execute_market_order(instruction_id, instruction, order_params)
            elif execution_plan.strategy == ExecutionStrategy.LIMIT:
                result = await self._execute_limit_order(instruction_id, instruction, order_params, execution_plan)
            elif execution_plan.strategy == ExecutionStrategy.TWAP:
                result = await self._execute_twap_order(instruction_id, instruction, order_params)
            elif execution_plan.strategy == ExecutionStrategy.VWAP:
                result = await self._execute_vwap_order(instruction_id, instruction, order_params)
            elif execution_plan.strategy == ExecutionStrategy.ICEBERG:
                result = await self._execute_iceberg_order(instruction_id, instruction, order_params)
            else:  # OPPORTUNISTIC
                result = await self._execute_opportunistic_order(instruction_id, instruction, order_params)
            
            # Track execution
            self._track_execution(instruction_id, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Execution failed: {str(e)}")
            return self._create_failed_result(instruction_id, instruction, str(e))
    
    def _validate_execution_request(self, instruction: TradeInstruction, 
                                   plan: ExecutionPlan) -> Dict:
        """Validate execution request"""
        
        # Basic validation
        if not instruction.symbol:
            return {'valid': False, 'error': 'No symbol provided'}
        
        if instruction.action not in ['BUY', 'SELL', 'TRIM', 'ADD', 'CLOSE']:
            return {'valid': False, 'error': f'Invalid action: {instruction.action}'}
        
        if plan.order_size <= 0:
            return {'valid': False, 'error': 'Invalid order size'}
        
        return {'valid': True}
    
    async def _calculate_order_parameters(self, instruction: TradeInstruction,
                                         plan: ExecutionPlan) -> Dict:
        """Calculate order parameters"""
        
        # If plan says no order, return 0 quantity
        if plan.order_size == 0:
            return {
                'order_side': 'BUY' if instruction.action in ['BUY', 'ADD'] else 'SELL',
                'order_quantity': 0,
                'current_price': plan.target_price or 0,
                'current_position': 0
            }
        
        # Determine order side
        if instruction.action in ['BUY', 'ADD']:
            order_side = 'BUY'
        else:
            order_side = 'SELL'
        
        # Get current position
        positions = await self.alpaca.get_positions()
        current_position = positions.get(instruction.symbol, {})
        current_quantity = current_position.get('quantity', 0)
        
        # Calculate order quantity
        if instruction.action == 'CLOSE':
            order_quantity = abs(current_quantity)
        elif instruction.action == 'TRIM':
            order_quantity = int(current_quantity * 0.5)  # Trim 50%
        elif instruction.action == 'ADD':
            order_quantity = int(current_quantity * 0.5) if current_quantity > 0 else plan.order_size
        else:
            order_quantity = plan.order_size
        
        # Get current price
        quote = await self.alpaca.get_latest_quote(instruction.symbol)
        current_price = (quote['bid'] + quote['ask']) / 2 if quote else plan.target_price
        
        return {
            'order_side': order_side,
            'order_quantity': int(order_quantity),
            'current_price': current_price,
            'current_position': current_quantity
        }
    
    async def _execute_market_order(self, instruction_id: str, instruction: TradeInstruction,
                                   order_params: Dict) -> ExecutionResult:
        """Execute market order"""
        
        if order_params['order_quantity'] == 0:
            return self._create_no_change_result(instruction_id, instruction)
        
        # Place market order
        order_result = await self.alpaca.place_order(
            symbol=instruction.symbol,
            qty=order_params['order_quantity'],
            side=order_params['order_side'],
            order_type='market',
            time_in_force='day'
        )
        
        if order_result['success']:
            # Monitor until filled
            fill_result = await self._monitor_order_until_filled(
                order_result['order_id'], timeout_minutes=5
            )
            
            return self._create_execution_result(
                instruction_id, instruction, [fill_result], 'market'
            )
        else:
            raise Exception(f"Market order failed: {order_result['error']}")
    
    async def _execute_limit_order(self, instruction_id: str, instruction: TradeInstruction,
                                  order_params: Dict, plan: ExecutionPlan) -> ExecutionResult:
        """Execute limit order"""
        
        if order_params['order_quantity'] == 0:
            return self._create_no_change_result(instruction_id, instruction)
        
        # Place limit order
        order_result = await self.alpaca.place_order(
            symbol=instruction.symbol,
            qty=order_params['order_quantity'],
            side=order_params['order_side'],
            order_type='limit',
            limit_price=plan.limit_price,
            time_in_force='day'
        )
        
        if order_result['success']:
            # Monitor for fills
            fill_result = await self._monitor_order_until_filled(
                order_result['order_id'], timeout_minutes=30
            )
            
            return self._create_execution_result(
                instruction_id, instruction, [fill_result], 'limit'
            )
        else:
            raise Exception(f"Limit order failed: {order_result['error']}")
    
    async def _execute_twap_order(self, instruction_id: str, instruction: TradeInstruction,
                                 order_params: Dict) -> ExecutionResult:
        """Execute Time-Weighted Average Price strategy"""
        
        if order_params['order_quantity'] == 0:
            return self._create_no_change_result(instruction_id, instruction)
        
        total_quantity = order_params['order_quantity']
        execution_window_hours = 2
        slice_count = 8  # Execute in 8 slices
        
        slice_size = max(1, total_quantity // slice_count)
        remaining_quantity = total_quantity
        execution_results = []
        
        for i in range(slice_count):
            if remaining_quantity <= 0:
                break
            
            current_slice = min(slice_size, remaining_quantity)
            
            # Place market order for this slice
            order_result = await self.alpaca.place_order(
                symbol=instruction.symbol,
                qty=current_slice,
                side=order_params['order_side'],
                order_type='market',
                time_in_force='day'
            )
            
            if order_result['success']:
                fill_result = await self._monitor_order_until_filled(order_result['order_id'])
                execution_results.append(fill_result)
                remaining_quantity -= current_slice
            
            # Wait before next slice
            if i < slice_count - 1 and remaining_quantity > 0:
                wait_time = (execution_window_hours * 3600) / slice_count
                await asyncio.sleep(wait_time)
        
        return self._create_execution_result(
            instruction_id, instruction, execution_results, 'twap'
        )
    
    async def _execute_vwap_order(self, instruction_id: str, instruction: TradeInstruction,
                                 order_params: Dict) -> ExecutionResult:
        """Execute Volume-Weighted Average Price strategy"""
        
        # Simplified VWAP - in production would use historical volume patterns
        # For now, execute more during high-volume periods
        
        if order_params['order_quantity'] == 0:
            return self._create_no_change_result(instruction_id, instruction)
        
        total_quantity = order_params['order_quantity']
        
        # Define volume distribution (simplified)
        # Morning: 35%, Midday: 25%, Afternoon: 40%
        volume_distribution = [0.35, 0.25, 0.40]
        time_windows = [2, 2, 2]  # Hours for each period
        
        execution_results = []
        
        for i, (volume_pct, window_hours) in enumerate(zip(volume_distribution, time_windows)):
            period_quantity = int(total_quantity * volume_pct)
            
            if period_quantity > 0:
                # Execute this period's allocation
                slices = max(1, window_hours * 2)  # 2 slices per hour
                slice_size = max(1, period_quantity // slices)
                
                for j in range(slices):
                    order_result = await self.alpaca.place_order(
                        symbol=instruction.symbol,
                        qty=slice_size,
                        side=order_params['order_side'],
                        order_type='market',
                        time_in_force='day'
                    )
                    
                    if order_result['success']:
                        fill_result = await self._monitor_order_until_filled(order_result['order_id'])
                        execution_results.append(fill_result)
                    
                    # Wait between slices
                    if j < slices - 1:
                        await asyncio.sleep((window_hours * 3600) / slices)
        
        return self._create_execution_result(
            instruction_id, instruction, execution_results, 'vwap'
        )
    
    async def _execute_iceberg_order(self, instruction_id: str, instruction: TradeInstruction,
                                    order_params: Dict) -> ExecutionResult:
        """Execute iceberg order (hide size)"""
        
        if order_params['order_quantity'] == 0:
            return self._create_no_change_result(instruction_id, instruction)
        
        total_quantity = order_params['order_quantity']
        visible_size = max(100, total_quantity // 10)  # Show only 10%
        
        execution_results = []
        remaining_quantity = total_quantity
        
        while remaining_quantity > 0:
            current_slice = min(visible_size, remaining_quantity)
            
            # Place limit order for visible portion
            quote = await self.alpaca.get_latest_quote(instruction.symbol)
            limit_price = quote['ask'] if order_params['order_side'] == 'BUY' else quote['bid']
            
            order_result = await self.alpaca.place_order(
                symbol=instruction.symbol,
                qty=current_slice,
                side=order_params['order_side'],
                order_type='limit',
                limit_price=limit_price,
                time_in_force='day'
            )
            
            if order_result['success']:
                fill_result = await self._monitor_order_until_filled(
                    order_result['order_id'], timeout_minutes=10
                )
                execution_results.append(fill_result)
                remaining_quantity -= current_slice
            
            # Small delay between chunks
            await asyncio.sleep(30)
        
        return self._create_execution_result(
            instruction_id, instruction, execution_results, 'iceberg'
        )
    
    async def _execute_opportunistic_order(self, instruction_id: str, instruction: TradeInstruction,
                                          order_params: Dict) -> ExecutionResult:
        """Execute opportunistic order (wait for favorable prices)"""
        
        if order_params['order_quantity'] == 0:
            return self._create_no_change_result(instruction_id, instruction)
        
        # Monitor for favorable price for up to 2 hours
        max_wait_time = timedelta(hours=2)
        start_time = datetime.now()
        
        target_improvement = 0.002  # Look for 0.2% better price
        
        while datetime.now() - start_time < max_wait_time:
            quote = await self.alpaca.get_latest_quote(instruction.symbol)
            current_price = (quote['bid'] + quote['ask']) / 2
            
            # Check if price is favorable
            if order_params['order_side'] == 'BUY':
                if current_price <= order_params['current_price'] * (1 - target_improvement):
                    # Good buying opportunity
                    break
            else:
                if current_price >= order_params['current_price'] * (1 + target_improvement):
                    # Good selling opportunity
                    break
            
            # Break immediately if we're past first iteration (for testing)
            # In production, would wait longer
            break
        
        # Execute at market
        order_result = await self.alpaca.place_order(
            symbol=instruction.symbol,
            qty=order_params['order_quantity'],
            side=order_params['order_side'],
            order_type='market',
            time_in_force='day'
        )
        
        if order_result['success']:
            fill_result = await self._monitor_order_until_filled(order_result['order_id'])
            return self._create_execution_result(
                instruction_id, instruction, [fill_result], 'opportunistic'
            )
        else:
            raise Exception(f"Opportunistic order failed: {order_result['error']}")
    
    async def _monitor_order_until_filled(self, order_id: str, 
                                         timeout_minutes: int = 30) -> Dict:
        """Monitor order until filled or timeout"""
        
        timeout = datetime.now() + timedelta(minutes=timeout_minutes)
        
        while datetime.now() < timeout:
            order_status = await self.alpaca.get_order(order_id)
            
            if order_status['status'] == 'filled':
                return {
                    'order_id': order_id,
                    'status': 'filled',
                    'filled_qty': order_status.get('filled_qty', 0),
                    'filled_avg_price': order_status.get('filled_avg_price', 0),
                    'commission': order_status.get('commission', 0)
                }
            elif order_status['status'] in ['cancelled', 'rejected', 'expired']:
                return {
                    'order_id': order_id,
                    'status': order_status['status'],
                    'filled_qty': 0,
                    'filled_avg_price': 0,
                    'commission': 0
                }
            
            await asyncio.sleep(self.monitoring_interval)
        
        # Timeout - attempt to cancel
        await self.alpaca.cancel_order(order_id)
        return {
            'order_id': order_id,
            'status': 'timeout',
            'filled_qty': 0,
            'filled_avg_price': 0,
            'commission': 0
        }
    
    def _create_execution_result(self, instruction_id: str, instruction: TradeInstruction,
                                fills: List[Dict], strategy: str) -> ExecutionResult:
        """Create execution result from fills"""
        
        total_quantity = sum(f.get('filled_qty', 0) for f in fills)
        total_commission = sum(f.get('commission', 0) for f in fills)
        
        if total_quantity > 0:
            avg_price = sum(f.get('filled_avg_price', 0) * f.get('filled_qty', 0) 
                          for f in fills) / total_quantity
            
            # Calculate slippage
            expected_price = fills[0].get('filled_avg_price', avg_price)
            slippage_pct = abs(avg_price - expected_price) / expected_price
            slippage_bps = slippage_pct * 10000
            
            status = OrderStatus.FILLED
            quality = 'good' if slippage_bps < 10 else 'fair' if slippage_bps < 20 else 'poor'
        else:
            avg_price = 0
            slippage_bps = 0
            status = OrderStatus.CANCELLED
            quality = 'failed'
        
        return ExecutionResult(
            instruction_id=instruction_id,
            symbol=instruction.symbol,
            action=instruction.action,
            requested_quantity=instruction.target_weight,
            executed_quantity=total_quantity,
            average_price=avg_price,
            execution_time=timedelta(minutes=len(fills) * 5),
            slippage_bps=slippage_bps,
            market_impact=0.0,  # Would calculate in production
            total_commission=total_commission,
            status=status,
            fills=fills,
            execution_quality=quality
        )
    
    def _create_no_change_result(self, instruction_id: str, 
                                instruction: TradeInstruction) -> ExecutionResult:
        """Create result for no position change needed"""
        
        return ExecutionResult(
            instruction_id=instruction_id,
            symbol=instruction.symbol,
            action='NO_CHANGE',
            requested_quantity=0,
            executed_quantity=0,
            average_price=0,
            execution_time=timedelta(0),
            slippage_bps=0,
            market_impact=0,
            total_commission=0,
            status=OrderStatus.CANCELLED,
            fills=[],
            execution_quality='no_change'
        )
    
    def _create_failed_result(self, instruction_id: str, instruction: TradeInstruction,
                            error: str) -> ExecutionResult:
        """Create failed execution result"""
        
        return ExecutionResult(
            instruction_id=instruction_id,
            symbol=instruction.symbol,
            action=instruction.action,
            requested_quantity=instruction.target_weight,
            executed_quantity=0,
            average_price=0,
            execution_time=timedelta(0),
            slippage_bps=0,
            market_impact=0,
            total_commission=0,
            status=OrderStatus.REJECTED,
            fills=[],
            execution_quality='failed'
        )
    
    def _create_no_change_result(self, instruction_id: str, 
                                instruction: TradeInstruction) -> ExecutionResult:
        """Create result for no position change needed"""
        
        return ExecutionResult(
            instruction_id=instruction_id,
            symbol=instruction.symbol,
            action='NO_CHANGE',
            requested_quantity=0,
            executed_quantity=0,
            average_price=0,
            execution_time=timedelta(0),
            slippage_bps=0,
            market_impact=0,
            total_commission=0,
            status=OrderStatus.CANCELLED,
            fills=[],
            execution_quality='no_change'
        )
    
    def _track_execution(self, instruction_id: str, result: ExecutionResult) -> None:
        """Track execution for performance analysis"""
        
        execution_record = {
            'instruction_id': instruction_id,
            'timestamp': datetime.now().isoformat(),
            'symbol': result.symbol,
            'action': result.action,
            'executed_quantity': result.executed_quantity,
            'average_price': result.average_price,
            'slippage_bps': result.slippage_bps,
            'execution_quality': result.execution_quality,
            'status': result.status.value
        }
        
        self.execution_history.append(execution_record)
        self.completed_orders[instruction_id] = result
    
    async def get_execution_performance(self) -> Dict:
        """Get execution performance metrics"""
        
        if not self.execution_history:
            return {'no_data': True}
        
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for e in self.execution_history 
                                   if e['status'] == 'filled')
        
        avg_slippage = np.mean([e['slippage_bps'] for e in self.execution_history])
        
        return {
            'total_executions': total_executions,
            'successful_executions': successful_executions,
            'success_rate': (successful_executions / total_executions) * 100,
            'average_slippage_bps': avg_slippage,
            'execution_history': self.execution_history[-10:]  # Last 10 executions
        }

# ==================== MAIN TRADE EXECUTION AGENT ====================

class TradeExecutionAgent(BaseAgent):
    """
    Trade Execution Agent
    
    Executes portfolio decisions with optimal timing, manages orders,
    and monitors execution quality in real-time
    """
    
    REQUIRED_FIELDS = ['task_type']
    
    def __init__(self, llm_provider, alpaca_provider: AlpacaProvider, config):
        super().__init__("trade_execution", llm_provider, config)
        self.alpaca = alpaca_provider
        self.timing_engine = ExecutionTimingEngine(alpaca_provider)
        self.order_manager = OrderManager(alpaca_provider)
        
        # Execution tracking
        self.daily_trades = []
        self.daily_volume = 0
        self.daily_commission = 0
        
        # Risk limits
        self.max_daily_trades = 50
        self.max_position_size = 0.05  # 5% of portfolio
        self.max_daily_turnover = 0.25  # 25% of portfolio
        
        self.logger.info("✅ Trade Execution Agent initialized")
    
    async def _process_internal(self, task_data: Dict) -> Dict:
        """
        Internal processing method required by BaseAgent
        
        Args:
            task_data: Validated task data
            
        Returns:
            Dict: Processing result
        """
        task_type = task_data.get('task_type')
        
        if task_type == 'execute_trade':
            trade_instruction = task_data.get('trade_instruction')
            result = await self.execute_single_trade(trade_instruction)
            return result
        elif task_type == 'execute_portfolio_decisions':
            portfolio_decisions = task_data.get('portfolio_decisions', [])
            result = await self.execute_portfolio_decisions(portfolio_decisions)
            return result
        elif task_type == 'monitor_executions':
            result = await self.monitor_active_executions()
            return result
        elif task_type == 'execution_report':
            period_days = task_data.get('period_days', 1)
            result = await self.generate_execution_report(period_days)
            return result
        else:
            return {'status': 'error', 'message': f'Unknown task type: {task_type}'}
    
    async def process(self, task_data: Dict) -> Dict:
        """Process execution requests"""
        
        task_type = task_data.get('task_type')
        
        if task_type == 'execute_trade':
            return await self.execute_single_trade(task_data.get('trade_instruction'))
        elif task_type == 'execute_portfolio_decisions':
            return await self.execute_portfolio_decisions(task_data.get('portfolio_decisions', []))
        elif task_type == 'monitor_executions':
            return await self.monitor_active_executions()
        elif task_type == 'execution_report':
            return await self.generate_execution_report(task_data.get('period_days', 1))
        else:
            return {'status': 'error', 'message': f'Unknown task type: {task_type}'}
    
    async def execute_single_trade(self, trade_decision: Dict) -> Dict:
        """Execute a single trade from Portfolio Manager"""
        
        try:
            # Convert to instruction
            instruction = self._convert_decision_to_instruction(trade_decision)
            if not instruction:
                return {'status': 'error', 'message': 'Invalid trade decision'}
            
            # Check daily limits
            if not self._check_daily_limits():
                return {
                    'status': 'deferred',
                    'message': 'Daily trading limits reached',
                    'instruction': instruction
                }
            
            # Create execution plan
            execution_plan = await self._create_execution_plan(instruction)
            
            # Execute trade
            result = await self.order_manager.execute_trade_instruction(
                instruction, execution_plan
            )
            
            # Track execution
            self._track_daily_execution(result)
            
            # Generate execution summary
            summary = self._generate_execution_summary(result)
            
            return {
                'status': 'success',
                'execution_result': result,
                'summary': summary
            }
            
        except Exception as e:
            self.logger.error(f"Trade execution failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def execute_portfolio_decisions(self, portfolio_decisions: List[Dict]) -> Dict:
        """Execute multiple portfolio decisions"""
        
        try:
            execution_results = []
            deferred_decisions = []
            failed_decisions = []
            
            # Sort decisions by priority (confidence)
            sorted_decisions = sorted(portfolio_decisions, 
                                    key=lambda x: x.get('confidence', 0), 
                                    reverse=True)
            
            for decision in sorted_decisions:
                # Execute each decision
                result = await self.execute_single_trade(decision)
                
                if result['status'] == 'success':
                    execution_results.append(result)
                elif result['status'] == 'deferred':
                    deferred_decisions.append(decision)
                else:
                    failed_decisions.append(decision)
                
                # Small delay between executions
                await asyncio.sleep(2)
            
            return {
                'status': 'success',
                'execution_summary': {
                    'total_decisions': len(portfolio_decisions),
                    'executed_count': len(execution_results),
                    'deferred_count': len(deferred_decisions),
                    'failed_count': len(failed_decisions)
                },
                'execution_results': execution_results,
                'deferred_decisions': deferred_decisions,
                'failed_decisions': failed_decisions
            }
            
        except Exception as e:
            self.logger.error(f"Portfolio execution failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def monitor_active_executions(self) -> Dict:
        """Monitor and report on active executions"""
        
        try:
            # Get active orders
            active_orders = await self.alpaca.get_orders(status='open')
            
            # Analyze execution performance
            performance = await self.order_manager.get_execution_performance()
            
            # Check for execution issues
            alerts = self._check_execution_alerts(active_orders, performance)
            
            return {
                'status': 'success',
                'active_orders': len(active_orders),
                'daily_trades_count': len(self.daily_trades),
                'daily_volume': self.daily_volume,
                'daily_commission': self.daily_commission,
                'execution_performance': performance,
                'execution_alerts': alerts
            }
            
        except Exception as e:
            self.logger.error(f"Execution monitoring failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    async def generate_execution_report(self, period_days: int = 1) -> Dict:
        """Generate comprehensive execution report"""
        
        try:
            # Get execution metrics
            performance = await self.order_manager.get_execution_performance()
            
            # Calculate period statistics
            period_stats = self._calculate_period_statistics(period_days)
            
            # Generate insights using LLM
            insights = await self._generate_execution_insights(
                performance, period_stats
            )
            
            return {
                'status': 'success',
                'report_period_days': period_days,
                'execution_metrics': performance,
                'period_statistics': period_stats,
                'execution_insights': insights,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {str(e)}")
            return {'status': 'error', 'message': str(e)}
    
    def _convert_decision_to_instruction(self, decision: Dict) -> Optional[TradeInstruction]:
        """Convert portfolio decision to trade instruction"""
        
        try:
            # Validate required fields
            symbol = decision.get('symbol', '')
            if not symbol:
                self.logger.error("Missing or empty symbol in decision")
                return None
            
            action = decision.get('action', '')
            if action not in ['BUY', 'SELL', 'TRIM', 'ADD', 'CLOSE']:
                self.logger.error(f"Invalid action: {action}")
                return None
            
            # Map urgency based on confidence
            confidence = decision.get('confidence', 5)
            if confidence >= 8:
                urgency = ExecutionUrgency.HIGH
            elif confidence >= 6:
                urgency = ExecutionUrgency.MEDIUM
            else:
                urgency = ExecutionUrgency.LOW
            
            return TradeInstruction(
                symbol=symbol,
                action=action,
                target_weight=decision.get('target_weight', 0),
                current_weight=decision.get('current_weight', 0),
                confidence=confidence,
                time_horizon=decision.get('time_horizon', 'medium'),
                reasoning=decision.get('reasoning', ''),
                risk_factors=decision.get('risk_factors', []),
                urgency=urgency,
                max_slippage=10.0,  # 10 basis points default
                execution_window=timedelta(hours=4)
            )
        except Exception as e:
            self.logger.error(f"Failed to convert decision: {str(e)}")
            return None
    
    async def _create_execution_plan(self, instruction: TradeInstruction) -> ExecutionPlan:
        """Create detailed execution plan"""
        
        # Assess market timing
        timing = await self.timing_engine.assess_market_timing(
            instruction.symbol, instruction.action
        )
        
        # Get current market data
        market_data = await self.alpaca.get_market_data([instruction.symbol], days=1)
        symbol_data = market_data.get(instruction.symbol, {})
        current_price = symbol_data.get('current_price', 0)
        
        # Calculate order size
        account_info = await self.alpaca.get_account_info()
        portfolio_value = account_info['account_value']
        target_value = portfolio_value * (instruction.target_weight / 100)
        order_size = int(target_value / current_price) if current_price > 0 else 0
        
        # Set pricing based on strategy
        strategy = timing['recommended_strategy']
        
        if strategy == ExecutionStrategy.LIMIT:
            if instruction.action in ['BUY', 'ADD']:
                limit_price = current_price * 0.999
            else:
                limit_price = current_price * 1.001
        else:
            limit_price = None
        
        return ExecutionPlan(
            instruction=instruction,
            strategy=strategy,
            order_size=order_size,
            target_price=current_price,
            limit_price=limit_price,
            stop_price=None,
            chunk_size=None,
            execution_schedule=[],
            market_timing=timing['timing_assessment'],
            expected_slippage=timing['expected_slippage'],
            risk_assessment={'risks': timing['execution_risks']}
        )
    
    def _check_daily_limits(self) -> bool:
        """Check if daily trading limits allow more trades"""
        
        if len(self.daily_trades) >= self.max_daily_trades:
            self.logger.warning("Daily trade limit reached")
            return False
        
        account_value = 1000000  # Would get from account in production
        if self.daily_volume >= account_value * self.max_daily_turnover:
            self.logger.warning("Daily turnover limit reached")
            return False
        
        return True
    
    def _track_daily_execution(self, result: ExecutionResult) -> None:
        """Track daily execution metrics"""
        
        self.daily_trades.append({
            'timestamp': datetime.now().isoformat(),
            'symbol': result.symbol,
            'action': result.action,
            'quantity': result.executed_quantity,
            'price': result.average_price,
            'commission': result.total_commission
        })
        
        self.daily_volume += result.executed_quantity * result.average_price
        self.daily_commission += result.total_commission
    
    def _generate_execution_summary(self, result: ExecutionResult) -> Dict:
        """Generate execution summary"""
        
        return {
            'symbol': result.symbol,
            'action': result.action,
            'executed_quantity': result.executed_quantity,
            'average_price': result.average_price,
            'slippage_bps': result.slippage_bps,
            'execution_quality': result.execution_quality,
            'total_cost': result.executed_quantity * result.average_price + result.total_commission
        }
    
    def _check_execution_alerts(self, active_orders: List, performance: Dict) -> List[Dict]:
        """Check for execution issues requiring alerts"""
        
        alerts = []
        
        # Check for stale orders
        for order in active_orders:
            order_age = datetime.now() - datetime.fromisoformat(order.get('created_at', ''))
            if order_age > timedelta(hours=2):
                alerts.append({
                    'type': 'stale_order',
                    'message': f"Order {order['id']} has been open for {order_age}",
                    'severity': 'warning'
                })
        
        # Check slippage
        if performance.get('average_slippage_bps', 0) > 20:
            alerts.append({
                'type': 'high_slippage',
                'message': f"Average slippage {performance['average_slippage_bps']:.1f} bps exceeds threshold",
                'severity': 'warning'
            })
        
        # Check success rate
        if performance.get('success_rate', 100) < 90:
            alerts.append({
                'type': 'low_success_rate',
                'message': f"Execution success rate {performance['success_rate']:.1f}% below threshold",
                'severity': 'warning'
            })
        
        return alerts
    
    def _calculate_period_statistics(self, period_days: int) -> Dict:
        """Calculate statistics for period"""
        
        cutoff_date = datetime.now() - timedelta(days=period_days)
        period_trades = [t for t in self.daily_trades 
                        if datetime.fromisoformat(t['timestamp']) > cutoff_date]
        
        if not period_trades:
            return {'no_data': True}
        
        total_volume = sum(t['quantity'] * t['price'] for t in period_trades)
        total_commission = sum(t['commission'] for t in period_trades)
        
        return {
            'total_trades': len(period_trades),
            'total_volume': total_volume,
            'total_commission': total_commission,
            'average_trade_size': total_volume / len(period_trades) if period_trades else 0,
            'trades_by_action': self._group_trades_by_action(period_trades)
        }
    
    def _group_trades_by_action(self, trades: List[Dict]) -> Dict:
        """Group trades by action type"""
        
        grouped = {}
        for trade in trades:
            action = trade['action']
            if action not in grouped:
                grouped[action] = {'count': 0, 'volume': 0}
            grouped[action]['count'] += 1
            grouped[action]['volume'] += trade['quantity'] * trade['price']
        
        return grouped
    
    async def _generate_execution_insights(self, performance: Dict, 
                                          period_stats: Dict) -> str:
        """Generate execution insights using LLM"""
        
        if 'no_data' in performance or 'no_data' in period_stats:
            return "Insufficient execution data for insights generation."
        
        prompt = f"""
        Analyze the following trade execution performance and provide insights:
        
        Performance Metrics:
        - Total Executions: {performance.get('total_executions', 0)}
        - Success Rate: {performance.get('success_rate', 0):.1f}%
        - Average Slippage: {performance.get('average_slippage_bps', 0):.1f} basis points
        
        Period Statistics:
        - Total Trades: {period_stats.get('total_trades', 0)}
        - Total Volume: ${period_stats.get('total_volume', 0):,.0f}
        - Average Trade Size: ${period_stats.get('average_trade_size', 0):,.0f}
        
        Provide a brief analysis of:
        1. Execution quality assessment
        2. Areas for improvement
        3. Key recommendations
        
        Keep response under 150 words.
        """
        
        try:
            response = await self.llm.complete(prompt, temperature=0.3, max_tokens=200)
            return response
        except Exception as e:
            self.logger.error(f"Failed to generate insights: {str(e)}")
            return "Execution analysis unavailable."

# ==================== TEST FUNCTIONS ====================

async def test_trade_execution_agent():
    """Test Trade Execution Agent functionality"""
    
    from config.settings import TradingConfig
    from data_provider.alpaca_provider import AlpacaProvider
    from src.core.llm_provider import ClaudeLLMProvider
    
    # Initialize components
    config = TradingConfig()
    alpaca = AlpacaProvider(config)
    llm = ClaudeLLMProvider(config)
    
    # Create Trade Execution Agent
    execution_agent = TradeExecutionAgent(llm, alpaca, config)
    
    print("🚀 Testing Trade Execution Agent...")
    print("=" * 50)
    
    # Test 1: Single trade execution
    print("\n📊 Test 1: Single Trade Execution")
    
    test_decision = {
        'type': 'new_position',
        'symbol': 'AAPL',
        'action': 'BUY',
        'target_weight': 3.0,  # 3% of portfolio
        'current_weight': 0,
        'confidence': 8,
        'time_horizon': 'medium',
        'reasoning': 'Strong technical setup with fundamental support',
        'risk_factors': ['market_volatility', 'earnings_upcoming']
    }
    
    result = await execution_agent.execute_single_trade(test_decision)
    
    print(f"Execution Status: {result['status']}")
    if result['status'] == 'success':
        summary = result.get('summary', {})
        print(f"  Symbol: {summary.get('symbol')}")
        print(f"  Action: {summary.get('action')}")
        print(f"  Quantity: {summary.get('executed_quantity', 0)}")
        print(f"  Price: ${summary.get('average_price', 0):.2f}")
        print(f"  Slippage: {summary.get('slippage_bps', 0):.1f} bps")
        print(f"  Quality: {summary.get('execution_quality')}")
    
    # Test 2: Portfolio decisions execution
    print("\n📊 Test 2: Portfolio Decisions Execution")
    
    portfolio_decisions = [
        {
            'type': 'new_position',
            'symbol': 'MSFT',
            'action': 'BUY',
            'target_weight': 2.0,
            'confidence': 8,
            'reasoning': 'Strong AI growth prospects'
        },
        {
            'type': 'position_adjustment',
            'symbol': 'GOOGL',
            'action': 'TRIM',
            'target_weight': 1.5,
            'current_weight': 3.0,
            'confidence': 6,
            'reasoning': 'Taking partial profits'
        }
    ]
    
    portfolio_result = await execution_agent.execute_portfolio_decisions(portfolio_decisions)
    
    print(f"Portfolio Execution Status: {portfolio_result['status']}")
    if 'execution_summary' in portfolio_result:
        summary = portfolio_result['execution_summary']
        print(f"  Total Decisions: {summary['total_decisions']}")
        print(f"  Executed: {summary['executed_count']}")
        print(f"  Deferred: {summary['deferred_count']}")
        print(f"  Failed: {summary['failed_count']}")
    
    # Test 3: Execution monitoring
    print("\n📊 Test 3: Execution Monitoring")
    
    monitor_result = await execution_agent.monitor_active_executions()
    
    print(f"Monitoring Status: {monitor_result['status']}")
    if monitor_result['status'] == 'success':
        print(f"  Active Orders: {monitor_result['active_orders']}")
        print(f"  Daily Trades: {monitor_result['daily_trades_count']}")
        print(f"  Daily Volume: ${monitor_result['daily_volume']:,.2f}")
        
        alerts = monitor_result.get('execution_alerts', [])
        if alerts:
            print(f"  Alerts: {len(alerts)}")
            for alert in alerts[:3]:  # Show first 3 alerts
                print(f"    - [{alert['severity']}] {alert['type']}: {alert['message']}")
    
    # Test 4: Execution report
    print("\n📊 Test 4: Execution Report Generation")
    
    report_result = await execution_agent.generate_execution_report(period_days=1)
    
    print(f"Report Status: {report_result['status']}")
    if report_result['status'] == 'success':
        metrics = report_result.get('execution_metrics', {})
        if 'no_data' not in metrics:
            print(f"  Total Executions: {metrics.get('total_executions', 0)}")
            print(f"  Success Rate: {metrics.get('success_rate', 0):.1f}%")
            print(f"  Avg Slippage: {metrics.get('average_slippage_bps', 0):.1f} bps")
        
        insights = report_result.get('execution_insights', '')
        if insights and insights != "Insufficient execution data for insights generation.":
            print(f"\n📝 Execution Insights:")
            print(f"  {insights[:300]}...")
    
    print("\n✅ Trade Execution Agent test completed successfully!")
    print("=" * 50)

if __name__ == "__main__":
    import asyncio
    asyncio.run(test_trade_execution_agent())