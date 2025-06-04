#!/usr/bin/env python3
"""
Team Test Runner for Crypto HFT Strategy

Allows each team member to run tests with their specific configuration
while maintaining consistent methodology and results comparison.
"""

import yaml
import sys
from pathlib import Path
import argparse
import logging
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from main import AsyncCryptoTradingPipeline

def load_team_config(member_id: int) -> dict:
    """Load configuration for specific team member."""
    config_file = project_root / f"team_configs/team_member_{member_id}.yaml"
    
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def run_team_test(member_id: int, output_suffix: str = None):
    """Run test for specific team member."""
    
    # Load team configuration
    config = load_team_config(member_id)
    test_params = config.pop('test_params', {})
    
    # Setup logging with team member info
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"team_member_{member_id}_{test_params.get('name', 'test')}_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"🚀 Starting test for Team Member {member_id}")
    logger.info(f"📋 Test name: {test_params.get('name', 'Standard Test')}")
    logger.info(f"📊 Max records: {test_params.get('max_records', 'unlimited')}")
    logger.info(f"⏱️  Time window: {test_params.get('time_window_ms', 100)}ms")
    
    # Save config to temporary file
    temp_config_file = f"temp_config_member_{member_id}.yaml"
    with open(temp_config_file, 'w') as f:
        yaml.dump(config, f, indent=2)
    
    try:
        # Initialize pipeline with team config
        pipeline = AsyncCryptoTradingPipeline(config_path=temp_config_file)
        
        # Run pipeline with team-specific parameters
        max_records = test_params.get('max_records')
        time_window = test_params.get('time_window_ms', 100)
        
        logger.info(f"🔄 Loading data (max_records={max_records}, time_window={time_window}ms)")
        pipeline.load_data()
        
        logger.info(f"🔄 Converting to event stream")
        pipeline.convert_to_event_stream()
        
        logger.info(f"🔄 Running backtest")
        results = pipeline.run_temporal_split_backtest()
        
        # Export results with team member suffix
        output_dir = f"results/team_member_{member_id}"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save detailed results
        results_file = f"{output_dir}/{test_params.get('name', 'test')}_{timestamp}.yaml"
        with open(results_file, 'w') as f:
            yaml.dump(results, f, indent=2, default_flow_style=False)
        
        logger.info(f"✅ Test completed successfully!")
        logger.info(f"📁 Results saved to: {output_dir}")
        
        # Print summary
        if results and 'out_of_sample_results' in results:
            oos = results['out_of_sample_results']
            logger.info(f"")
            logger.info(f"🎯 TEAM MEMBER {member_id} RESULTS SUMMARY:")
            logger.info(f"Strategy: {test_params.get('name', 'Standard')}")
            logger.info(f"Net Return: {oos.get('net_return', 0):.2%}")
            logger.info(f"Sharpe Ratio: {oos.get('sharpe_ratio', 0):.3f}")
            logger.info(f"Max Drawdown: {oos.get('max_drawdown', 0):.2%}")
            logger.info(f"Total Trades: {oos.get('num_trades', 0)}")
            logger.info(f"Transaction Costs: ${oos.get('total_transaction_costs', 0):,.2f}")
            logger.info(f"")
            
        return results
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    finally:
        # Cleanup
        if Path(temp_config_file).exists():
            Path(temp_config_file).unlink()

def compare_team_results():
    """Compare results from all team members."""
    results_dir = Path("results")
    team_results = {}
    
    for member_dir in results_dir.glob("team_member_*"):
        if member_dir.is_dir():
            member_id = int(member_dir.name.split("_")[-1])
            
            # Find latest result file
            result_files = list(member_dir.glob("*.yaml"))
            if result_files:
                latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
                
                with open(latest_file, 'r') as f:
                    team_results[member_id] = yaml.safe_load(f)
    
    # Print comparison
    print("\n" + "="*80)
    print("🏆 TEAM RESULTS COMPARISON")
    print("="*80)
    
    for member_id in sorted(team_results.keys()):
        results = team_results[member_id]
        oos = results.get('out_of_sample_results', {})
        
        print(f"\n👤 Team Member {member_id}:")
        print(f"   Net Return: {oos.get('net_return', 0):>8.2%}")
        print(f"   Sharpe:     {oos.get('sharpe_ratio', 0):>8.3f}")
        print(f"   Max DD:     {oos.get('max_drawdown', 0):>8.2%}")
        print(f"   Trades:     {oos.get('num_trades', 0):>8}")
        print(f"   Costs:      ${oos.get('total_transaction_costs', 0):>7,.0f}")

def main():
    parser = argparse.ArgumentParser(description="Team Test Runner for Crypto HFT")
    parser.add_argument("--member", type=int, choices=range(1, 7), 
                       help="Team member ID (1-6)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare results from all team members")
    parser.add_argument("--all", action="store_true",
                       help="Run tests for all team members")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_team_results()
    elif args.all:
        print("🚀 Running tests for all team members...")
        for member_id in range(1, 7):
            print(f"\n{'='*60}")
            print(f"Starting Team Member {member_id}")
            print("="*60)
            try:
                run_team_test(member_id)
            except Exception as e:
                print(f"❌ Team Member {member_id} failed: {e}")
                continue
        
        print("\n🏁 All tests completed! Running comparison...")
        compare_team_results()
    elif args.member:
        run_team_test(args.member)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
