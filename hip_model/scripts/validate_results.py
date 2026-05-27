def validate_ce_angles(angles_df):
    """验证CE角度计算结果"""
    # 正常范围检查
    abnormal = angles_df[
        (angles_df['left_ce'] < 20) | 
        (angles_df['left_ce'] > 40) |
        (angles_df['right_ce'] < 20) | 
        (angles_df['right_ce'] > 40)
    ]
    
    if not abnormal.empty:
        print("\n=== 需要复查的病例 ===")
        for _, row in abnormal.iterrows():
            print(f"图片ID: {row['image_id']}")
            print(f"左侧CE角: {row['left_ce']:.1f}°")
            print(f"右侧CE角: {row['right_ce']:.1f}°\n") 