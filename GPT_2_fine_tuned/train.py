    axes[3].bar(range(len(keys)), values, tick_label=[f"{key[0]}-{key[1]}" for key in keys])
    axes[3].grid(axis="y")
    axes[3].set_xlabel("Temperatures Groups")
    axes[3].set_ylabel("Percentage")
    axes[3].set_title("Group Percentage")
    axes[3].set_ylim(0,1)

axes[3].set_xticklabels([f"{key[0]}-{key[1]}" for key in keys], rotation=20)

