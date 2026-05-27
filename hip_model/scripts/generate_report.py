def generate_report(results_df):
    """生成分析报告"""
    from datetime import datetime
    
    report = f"""
    CE角度测量报告
    生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}
    
    总计分析图片: {len(results_df)} 张
    
    统计结果:
    - 左侧CE角度: {results_df['left_ce'].mean():.1f}° ± {results_df['left_ce'].std():.1f}°
    - 右侧CE角度: {results_df['right_ce'].mean():.1f}° ± {results_df['right_ce'].std():.1f}°
    
    异常病例数: {len(results_df[
        (results_df['left_ce'] < 20) | 
        (results_df['right_ce'] < 20)
    ])}
    """
    
    with open('ce_angle_report.txt', 'w') as f:
        f.write(report) 