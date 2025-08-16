# utils/markdown_reporter.py
"""
Markdown Report Generator
Replaces Telegram notifications with markdown reports for macOS development
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

class MarkdownReporter:
    """
    Generate comprehensive markdown reports for trading system
    Replaces Telegram with file-based reporting for development
    """
    
    def __init__(self, reports_dir: str = "./reports"):
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger('markdown_reporter')
        
        # Create subdirectories
        (self.reports_dir / "daily").mkdir(exist_ok=True)
        (self.reports_dir / "trades").mkdir(exist_ok=True)
        (self.reports_dir / "analysis").mkdir(exist_ok=True)
        (self.reports_dir / "system").mkdir(exist_ok=True)
    
    def generate_daily_report(self, data: Dict[str, Any]) -> str:
        """Generate comprehensive daily trading report"""
        
        timestamp = datetime.now()
        date_str = timestamp.strftime("%Y-%m-%d")
        
        # Create report content
        report = self._create_daily_report_content(data, timestamp)
        
        # Save to file
        filename = f"daily_report_{date_str}.md"
        filepath = self.reports_dir / "daily" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Daily report generated: {filepath}")
        return str(filepath)
    
    def generate_trade_report(self, trade_data: Dict[str, Any]) -> str:
        """Generate individual trade execution report"""
        
        timestamp = datetime.now()
        trade_id = trade_data.get('trade_id', 'unknown')
        
        # Create report content
        report = self._create_trade_report_content(trade_data, timestamp)
        
        # Save to file
        filename = f"trade_{trade_id}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.reports_dir / "trades" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Trade report generated: {filepath}")
        return str(filepath)
    
    def generate_analysis_report(self, analysis_data: Dict[str, Any], 
                               report_type: str = "screening") -> str:
        """Generate analysis report (screening, portfolio, etc.)"""
        
        timestamp = datetime.now()
        
        # Create report content
        if report_type == "screening":
            report = self._create_screening_report_content(analysis_data, timestamp)
        elif report_type == "portfolio":
            report = self._create_portfolio_report_content(analysis_data, timestamp)
        else:
            report = self._create_generic_analysis_report(analysis_data, timestamp, report_type)
        
        # Save to file
        filename = f"{report_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.reports_dir / "analysis" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"Analysis report generated: {filepath}")
        return str(filepath)
    
    def generate_system_status_report(self, status_data: Dict[str, Any]) -> str:
        """Generate system health and status report"""
        
        timestamp = datetime.now()
        
        # Create report content
        report = self._create_system_status_report_content(status_data, timestamp)
        
        # Save to file
        filename = f"system_status_{timestamp.strftime('%Y%m%d_%H%M%S')}.md"
        filepath = self.reports_dir / "system" / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"System status report generated: {filepath}")
        return str(filepath)
    
    def _create_daily_report_content(self, data: Dict[str, Any], timestamp: datetime) -> str:
        """Create daily report markdown content"""
        
        return f"""# 📊 Daily Trading Report
**Date**: {timestamp.strftime('%B %d, %Y')}  
**Generated**: {timestamp.strftime('%I:%M %p')}

---

## 📈 Portfolio Summary

| Metric | Value |
|--------|-------|
| **Portfolio Value** | ${data.get('portfolio_value', 0):,.2f} |
| **Cash Available** | ${data.get('cash_available', 0):,.2f} |
| **Day P&L** | ${data.get('day_pnl', 0):,.2f} |
| **Total P&L** | ${data.get('total_pnl', 0):,.2f} |
| **Active Positions** | {data.get('active_positions', 0)} |

---

## 🎯 Today's Activity

### Trades Executed
{self._format_trades_section(data.get('trades', []))}

### New Positions
{self._format_positions_section(data.get('new_positions', []))}

### Closed Positions  
{self._format_positions_section(data.get('closed_positions', []))}

---

## 🔍 Market Analysis

### Screening Results
{self._format_screening_section(data.get('screening_results', {}))}

### Junior Analyst Insights
{self._format_analysis_section(data.get('junior_analysis', {}))}

### Senior Analyst Recommendations
{self._format_analysis_section(data.get('senior_analysis', {}))}

---

## ⚠️ Risk Management

| Risk Metric | Current | Limit | Status |
|-------------|---------|-------|--------|
| **Portfolio Drawdown** | {data.get('current_drawdown', 0)*100:.1f}% | {data.get('max_drawdown', 5)*100:.1f}% | {'🟢 OK' if data.get('current_drawdown', 0) < data.get('max_drawdown', 0.05) else '🔴 ALERT'} |
| **Position Concentration** | {data.get('max_position_size', 0)*100:.1f}% | {data.get('position_limit', 5)*100:.1f}% | {'🟢 OK' if data.get('max_position_size', 0) < data.get('position_limit', 0.05) else '🔴 ALERT'} |
| **Daily Trades** | {data.get('daily_trades', 0)} | {data.get('max_daily_trades', 10)} | {'🟢 OK' if data.get('daily_trades', 0) < data.get('max_daily_trades', 10) else '🔴 ALERT'} |

---

## 🤖 System Performance

### Agent Status
{self._format_agent_status(data.get('agent_status', {}))}

### API Usage
{self._format_api_usage(data.get('api_usage', {}))}

---

## 📋 Action Items

{self._format_action_items(data.get('action_items', []))}

---

## 📊 Charts and Data

### Performance Chart
```
{self._create_simple_ascii_chart(data.get('performance_data', []))}
```

---

*Report generated by AI Trading System v1.0*  
*Last updated: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}*
"""

    def _create_trade_report_content(self, trade_data: Dict[str, Any], timestamp: datetime) -> str:
        """Create trade execution report content"""
        
        return f"""# 💼 Trade Execution Report
**Trade ID**: {trade_data.get('trade_id', 'N/A')}  
**Timestamp**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📋 Trade Details

| Field | Value |
|-------|-------|
| **Symbol** | {trade_data.get('symbol', 'N/A')} |
| **Side** | {'🟢 BUY' if trade_data.get('side', '').upper() == 'BUY' else '🔴 SELL'} |
| **Quantity** | {trade_data.get('quantity', 0):,.0f} shares |
| **Order Type** | {trade_data.get('order_type', 'N/A').upper()} |
| **Limit Price** | ${trade_data.get('limit_price', 0):,.2f} |
| **Filled Price** | ${trade_data.get('filled_price', 0):,.2f} |
| **Total Value** | ${trade_data.get('total_value', 0):,.2f} |

---

## 📊 Execution Analysis

### Performance Metrics
- **Slippage**: ${trade_data.get('slippage', 0):,.4f} ({trade_data.get('slippage_percent', 0)*100:.2f}%)
- **Fill Time**: {trade_data.get('fill_time', 'N/A')} seconds
- **Execution Quality**: {'🟢 GOOD' if trade_data.get('slippage_percent', 0) < 0.001 else '🟡 FAIR' if trade_data.get('slippage_percent', 0) < 0.005 else '🔴 POOR'}

### Market Conditions
- **Market Status**: {trade_data.get('market_status', 'Unknown')}
- **Bid/Ask Spread**: ${trade_data.get('spread', 0):,.4f}
- **Volume**: {trade_data.get('volume', 0):,} shares

---

## 🎯 Trade Rationale

### Analysis Summary
{trade_data.get('analysis_summary', 'No analysis provided')}

### Risk Assessment
- **Stop Loss**: ${trade_data.get('stop_loss', 0):,.2f}
- **Take Profit**: ${trade_data.get('take_profit', 0):,.2f}
- **Risk/Reward**: {trade_data.get('risk_reward_ratio', 'N/A')}

---

## 📈 Position Impact

### Portfolio Changes
- **Portfolio Value Before**: ${trade_data.get('portfolio_before', 0):,.2f}
- **Portfolio Value After**: ${trade_data.get('portfolio_after', 0):,.2f}
- **Position Size**: {trade_data.get('position_percent', 0)*100:.1f}% of portfolio

---

*Trade report generated by AI Trading System*
"""

    def _create_screening_report_content(self, data: Dict[str, Any], timestamp: datetime) -> str:
        """Create screening analysis report content"""
        
        return f"""# 🔍 Technical Screening Report
**Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Screening Summary

### Overview
- **Symbols Screened**: {data.get('symbols_screened', 0)}
- **Total Screens**: {data.get('total_screens', 0)}
- **Successful Screens**: {data.get('successful_screens', 0)}
- **Success Rate**: {(data.get('successful_screens', 0) / max(data.get('total_screens', 1), 1)) * 100:.1f}%

---

## 🎯 Top Opportunities

{self._format_screening_opportunities(data.get('top_opportunities', []))}

---

## 📋 Detailed Results

{self._format_detailed_screening_results(data.get('screens', {}))}

---

## 📈 Pattern Analysis

### Bullish Patterns
{self._format_pattern_list(data.get('bullish_patterns', []))}

### Bearish Patterns  
{self._format_pattern_list(data.get('bearish_patterns', []))}

### Neutral Patterns
{self._format_pattern_list(data.get('neutral_patterns', []))}

---

*Screening report generated by AI Trading System*
"""

    def _create_portfolio_report_content(self, data: Dict[str, Any], timestamp: datetime) -> str:
        """Create portfolio analysis report content"""
        
        return f"""# 💼 Portfolio Analysis Report
**Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 📊 Portfolio Overview

### Current Holdings
{self._format_portfolio_holdings(data.get('holdings', []))}

### Performance Metrics
{self._format_performance_metrics(data.get('performance', {}))}

### Risk Analysis
{self._format_risk_analysis(data.get('risk_analysis', {}))}

---

## 🎯 Recommendations

{self._format_recommendations(data.get('recommendations', []))}

---

*Portfolio analysis generated by AI Trading System*
"""

    def _create_system_status_report_content(self, data: Dict[str, Any], timestamp: datetime) -> str:
        """Create system status report content"""
        
        return f"""# 🖥️ System Status Report
**Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## 🟢 System Health

### Overall Status
- **System Status**: {'🟢 HEALTHY' if data.get('healthy', True) else '🔴 ISSUES DETECTED'}
- **Uptime**: {data.get('uptime', 'Unknown')}
- **Last Error**: {data.get('last_error', 'None')}

### API Connections
{self._format_api_status(data.get('api_status', {}))}

### Agent Status
{self._format_agent_health(data.get('agents', {}))}

---

## 📊 Performance Metrics

### Resource Usage
- **CPU Usage**: {data.get('cpu_usage', 0):.1f}%
- **Memory Usage**: {data.get('memory_usage', 0):.1f}%
- **Disk Usage**: {data.get('disk_usage', 0):.1f}%

### API Usage
{self._format_detailed_api_usage(data.get('api_usage', {}))}

---

*System status generated by AI Trading System*
"""

    def _create_generic_analysis_report(self, data: Dict[str, Any], timestamp: datetime, report_type: str) -> str:
        """Create generic analysis report"""
        
        return f"""# 📊 {report_type.title()} Analysis Report
**Generated**: {timestamp.strftime('%Y-%m-%d %H:%M:%S')}

---

## Analysis Results

{json.dumps(data, indent=2, default=str)}

---

*Analysis report generated by AI Trading System*
"""

    # Helper formatting methods
    def _format_trades_section(self, trades: List[Dict]) -> str:
        if not trades:
            return "No trades executed today."
        
        result = "| Symbol | Side | Quantity | Price | Total |\n|--------|------|----------|-------|-------|\n"
        for trade in trades[:10]:  # Limit to last 10
            result += f"| {trade.get('symbol', 'N/A')} | {trade.get('side', 'N/A')} | {trade.get('quantity', 0):,.0f} | ${trade.get('price', 0):,.2f} | ${trade.get('total', 0):,.2f} |\n"
        return result

    def _format_positions_section(self, positions: List[Dict]) -> str:
        if not positions:
            return "No new positions."
        
        result = "| Symbol | Quantity | Value | P&L |\n|--------|----------|-------|-----|\n"
        for pos in positions[:10]:
            result += f"| {pos.get('symbol', 'N/A')} | {pos.get('quantity', 0):,.0f} | ${pos.get('value', 0):,.2f} | ${pos.get('pnl', 0):,.2f} |\n"
        return result

    def _format_screening_section(self, results: Dict) -> str:
        if not results:
            return "No screening results available."
        
        summary = results.get('summary', {})
        return f"""
**Total Opportunities**: {summary.get('total_opportunities', 0)}  
**Momentum Signals**: {summary.get('momentum_signals', 0)}  
**Oversold Conditions**: {summary.get('oversold_conditions', 0)}  
**Volume Spikes**: {summary.get('volume_spikes', 0)}
"""

    def _format_analysis_section(self, analysis: Dict) -> str:
        if not analysis:
            return "No analysis available."
        
        return f"""
**Stocks Analyzed**: {analysis.get('stocks_analyzed', 0)}  
**Buy Recommendations**: {analysis.get('buy_recommendations', 0)}  
**Sell Recommendations**: {analysis.get('sell_recommendations', 0)}  
**Hold Recommendations**: {analysis.get('hold_recommendations', 0)}
"""

    def _format_agent_status(self, status: Dict) -> str:
        if not status:
            return "No agent status available."
        
        result = "| Agent | Status | Last Run |\n|-------|--------|----------|\n"
        for agent, info in status.items():
            status_emoji = "🟢" if info.get('healthy', True) else "🔴"
            result += f"| {agent} | {status_emoji} {info.get('status', 'Unknown')} | {info.get('last_run', 'Unknown')} |\n"
        return result

    def _format_api_usage(self, usage: Dict) -> str:
        if not usage:
            return "No API usage data available."
        
        result = "| API | Calls | Limit | Usage % |\n|-----|-------|-------|--------|\n"
        for api, data in usage.items():
            calls = data.get('calls', 0)
            limit = data.get('limit', 1)
            percentage = (calls / limit) * 100 if limit > 0 else 0
            result += f"| {api} | {calls} | {limit} | {percentage:.1f}% |\n"
        return result

    def _format_action_items(self, items: List[str]) -> str:
        if not items:
            return "No action items."
        
        result = ""
        for i, item in enumerate(items, 1):
            result += f"{i}. {item}\n"
        return result

    def _create_simple_ascii_chart(self, data: List[float]) -> str:
        if not data or len(data) < 2:
            return "Insufficient data for chart"
        
        # Simple ASCII chart representation
        min_val = min(data)
        max_val = max(data)
        range_val = max_val - min_val if max_val != min_val else 1
        
        chart = ""
        for value in data[-20:]:  # Last 20 data points
            normalized = int(((value - min_val) / range_val) * 10)
            chart += "█" * normalized + "\n"
        
        return chart

    def _format_screening_opportunities(self, opportunities: List[Dict]) -> str:
        if not opportunities:
            return "No opportunities found."
        
        result = "| Symbol | Screens | Confidence |\n|--------|---------|------------|\n"
        for opp in opportunities[:10]:
            result += f"| {opp.get('symbol', 'N/A')} | {opp.get('screen_count', 0)} | {opp.get('confidence', 0)*100:.0f}% |\n"
        return result

    def _format_detailed_screening_results(self, screens: Dict) -> str:
        if not screens:
            return "No detailed results available."
        
        result = ""
        for screen_name, screen_data in screens.items():
            matches = screen_data.get('matches', [])
            result += f"\n### {screen_name.replace('_', ' ').title()}\n"
            result += f"**Matches**: {len(matches)}\n\n"
            if matches:
                result += "| Symbol | Details |\n|--------|--------|\n"
                for match in matches[:5]:  # Top 5 matches
                    symbol = match.get('symbol', 'N/A')
                    details = ", ".join([f"{k}: {v}" for k, v in match.items() if k != 'symbol'])
                    result += f"| {symbol} | {details} |\n"
        return result

    def _format_pattern_list(self, patterns: List[str]) -> str:
        if not patterns:
            return "No patterns detected."
        return "\n".join([f"- {pattern}" for pattern in patterns])

    def _format_portfolio_holdings(self, holdings: List[Dict]) -> str:
        if not holdings:
            return "No holdings."
        
        result = "| Symbol | Quantity | Value | Weight | P&L |\n|--------|----------|-------|--------|-----|\n"
        for holding in holdings:
            result += f"| {holding.get('symbol', 'N/A')} | {holding.get('quantity', 0):,.0f} | ${holding.get('value', 0):,.2f} | {holding.get('weight', 0)*100:.1f}% | ${holding.get('pnl', 0):,.2f} |\n"
        return result

    def _format_performance_metrics(self, performance: Dict) -> str:
        if not performance:
            return "No performance data."
        
        return f"""
| Metric | Value |
|--------|-------|
| **Daily Return** | {performance.get('daily_return', 0)*100:.2f}% |
| **Total Return** | {performance.get('total_return', 0)*100:.2f}% |
| **Sharpe Ratio** | {performance.get('sharpe_ratio', 0):.2f} |
| **Max Drawdown** | {performance.get('max_drawdown', 0)*100:.2f}% |
| **Win Rate** | {performance.get('win_rate', 0)*100:.1f}% |
"""

    def _format_risk_analysis(self, risk: Dict) -> str:
        if not risk:
            return "No risk analysis."
        
        return f"""
| Risk Metric | Value | Status |
|-------------|-------|--------|
| **VaR (95%)** | ${risk.get('var_95', 0):,.2f} | {'🟢 OK' if risk.get('var_95', 0) < 10000 else '🔴 HIGH'} |
| **Beta** | {risk.get('beta', 0):.2f} | {'🟢 LOW' if abs(risk.get('beta', 0)) < 1 else '🔴 HIGH'} |
| **Volatility** | {risk.get('volatility', 0)*100:.1f}% | {'🟢 LOW' if risk.get('volatility', 0) < 0.2 else '🔴 HIGH'} |
"""

    def _format_recommendations(self, recommendations: List[Dict]) -> str:
        if not recommendations:
            return "No recommendations."
        
        result = ""
        for i, rec in enumerate(recommendations, 1):
            result += f"\n### Recommendation {i}: {rec.get('action', 'N/A').upper()}\n"
            result += f"**Symbol**: {rec.get('symbol', 'N/A')}\n"
            result += f"**Rationale**: {rec.get('rationale', 'No rationale provided')}\n"
            result += f"**Confidence**: {rec.get('confidence', 0)*100:.0f}%\n"
        
        return result

    def _format_api_status(self, api_status: Dict) -> str:
        if not api_status:
            return "No API status available."
        
        result = "| API | Status | Response Time |\n|-----|--------|---------------|\n"
        for api, status in api_status.items():
            status_emoji = "🟢" if status.get('healthy', True) else "🔴"
            result += f"| {api} | {status_emoji} {status.get('status', 'Unknown')} | {status.get('response_time', 'N/A')}ms |\n"
        return result

    def _format_agent_health(self, agents: Dict) -> str:
        if not agents:
            return "No agent information available."
        
        result = "| Agent | Status | Last Activity | Errors |\n|-------|--------|---------------|--------|\n"
        for agent, info in agents.items():
            status_emoji = "🟢" if info.get('healthy', True) else "🔴"
            result += f"| {agent} | {status_emoji} {info.get('status', 'Unknown')} | {info.get('last_activity', 'Unknown')} | {info.get('error_count', 0)} |\n"
        return result

    def _format_detailed_api_usage(self, usage: Dict) -> str:
        if not usage:
            return "No API usage data."
        
        result = "| API | Calls Today | Rate Limit | Last Call | Errors |\n|-----|-------------|------------|-----------|--------|\n"
        for api, data in usage.items():
            result += f"| {api} | {data.get('calls_today', 0)} | {data.get('rate_limit', 'Unknown')} | {data.get('last_call', 'Unknown')} | {data.get('errors', 0)} |\n"
        return result

    # macOS specific methods
    def send_macos_notification(self, title: str, message: str):
        """Send macOS system notification"""
        try:
            os.system(f"""
                osascript -e 'display notification "{message}" with title "{title}"'
            """)
        except Exception as e:
            self.logger.error(f"Failed to send macOS notification: {e}")

    def open_latest_report(self, report_type: str = "daily"):
        """Open latest report in default markdown viewer"""
        try:
            report_dir = self.reports_dir / report_type
            if report_dir.exists():
                # Find latest report
                reports = list(report_dir.glob("*.md"))
                if reports:
                    latest_report = max(reports, key=os.path.getctime)
                    os.system(f"open '{latest_report}'")
        except Exception as e:
            self.logger.error(f"Failed to open report: {e}")


# Usage example and testing
if __name__ == "__main__":
    # Test the markdown reporter
    reporter = MarkdownReporter()
    
    # Sample data
    sample_daily_data = {
        'portfolio_value': 100000,
        'cash_available': 5000,
        'day_pnl': 1250,
        'total_pnl': 8500,
        'active_positions': 8,
        'trades': [
            {'symbol': 'AAPL', 'side': 'BUY', 'quantity': 100, 'price': 150.25, 'total': 15025},
            {'symbol': 'MSFT', 'side': 'SELL', 'quantity': 50, 'price': 350.50, 'total': 17525}
        ],
        'screening_results': {
            'summary': {
                'total_opportunities': 15,
                'momentum_signals': 8,
                'oversold_conditions': 3,
                'volume_spikes': 4
            }
        },
        'agent_status': {
            'junior_analyst': {'healthy': True, 'status': 'Active', 'last_run': '10:30 AM'},
            'senior_analyst': {'healthy': True, 'status': 'Active', 'last_run': '10:35 AM'},
            'portfolio_manager': {'healthy': True, 'status': 'Active', 'last_run': '10:40 AM'}
        }
    }
    
    # Generate test report
    report_path = reporter.generate_daily_report(sample_daily_data)
    print(f"✅ Test report generated: {report_path}")
    
    # Send macOS notification
    reporter.send_macos_notification("Trading System", "Daily report generated successfully!")
    
    print("🍎 Check your macOS notifications and the reports/ directory!")