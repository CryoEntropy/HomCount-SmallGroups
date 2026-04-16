# 辅助函数：CSV 转义
EscapeCSV := function(s)
  local str;
  str := String(s);
  if ForAny(str, c -> c in ",\"\n\r") then
    str := ReplacedString(str, "\"", "\"\"");
    return Concatenation("\"", str, "\"");
  else
    return str;
  fi;
end;

SimpleHomCount := function(G, H)
    local all_homs, invG, invH, count, i, j;
    
    # 如果H是无限群，直接返回无限
    if not IsFinite(H) then
        return "infinite";
    fi;
    
    # 如果G和H都是有限交换群，使用公式计算
    if IsFinite(G) and IsAbelian(G) and IsAbelian(H) then
        # 获取两个群的不变因子（初等因子）
        invG := AbelianInvariants(G);
        invH := AbelianInvariants(H);
        
        # 计算同态个数公式：∏_{i=1}^r ∏_{j=1}^s gcd(a_i, b_j)
        # 其中invG = [a_1, ..., a_r], invH = [b_1, ..., b_s]
        count := 1;
        for i in invG do
            for j in invH do
                count := count * Gcd(i, j);
            od;
        od;
        
        return count;
    else
        # 至少有一个不是交换群，使用原来的方法
        all_homs := AllHomomorphisms(G, H);
        return Length(all_homs);
    fi;
end;

# 计算同态数量矩阵，输出 CSV 表格（行=源群，列=靶群）
CrossOrderHomCounts := function(orderG, orderH, arg...)
  local mG, mH, i, j, Glist, Hlist, descG, descH, idsG, idsH, hom,
        outfile, fp, parts, colLabel;

  if not IsBound(SimpleHomCount) then
    Error("CrossOrderHomCounts: SimpleHomCount is not defined. Please define SimpleHomCount(G,H) first.");
  fi;

  # 解析可选参数 outfile（true / false / 字符串），默认 true（自动生成文件名）
  if Length(arg) = 0 then
    outfile := true;
  else
    outfile := arg[1];
  fi;

  mG := NumberSmallGroups(orderG);
  if mG = fail then
    Print("CrossOrderHomCounts: SmallGroups database does not contain groups of order ", orderG, ".\n");
    return;
  fi;
  mH := NumberSmallGroups(orderH);
  if mH = fail then
    Print("CrossOrderHomCounts: SmallGroups database does not contain groups of order ", orderH, ".\n");
    return;
  fi;

  Print("CrossOrderHomCounts: There are ", mG, " groups of order ", orderG,
        " and ", mH, " groups of order ", orderH, ".\n");
  Print("Note: will perform ", mG*mH, " SimpleHomCount calls.\n\n");

  Glist := List([1..mG], k -> SmallGroup(orderG, k));
  descG := List(Glist, G -> StructureDescription(G));
  idsG  := List(Glist, G -> IdGroup(G));

  Hlist := List([1..mH], k -> SmallGroup(orderH, k));
  descH := List(Hlist, H -> StructureDescription(H));
  idsH  := List(Hlist, H -> IdGroup(H));

  # 文件名处理
  if outfile = true then
    outfile := Concatenation("Hom_counts_of_cluster_", String(orderG), "_to_", String(orderH), ".csv");
  fi;

  # 如果 outfile 不是 false，则打开文件
  if outfile <> false then
    fp := OutputTextFile(outfile, false);
    SetPrintFormattingStatus(fp, false);
  else
    fp := fail;
  fi;

  Print("Output style: matrix (rows = source groups, cols = target groups)\n");

  # 写入列标题行（靶群标签）—— 只显示编号
  if outfile <> false then
    parts := [""];   # 第一列留空，用于行标签
    for j in [1..mH] do
      Add(parts, EscapeCSV(String(idsH[j][2])));   # 只输出第二个分量（编号）
    od;
    WriteLine(fp, JoinStringsWithSeparator(parts, ","));
  fi;

  # 双重循环计算
  for i in [1..mG] do
    # 行标签 —— 只显示源群编号
    parts := [EscapeCSV(String(idsG[i][2]))];

    for j in [1..mH] do
      hom := SimpleHomCount(Glist[i], Hlist[j]);

      # 屏幕打印（仍保留完整描述，便于调试）
      Print("|Hom(", idsG[i], " ", descG[i], " -> ",
                  idsH[j], " ", descH[j], ")| = ", hom, "\n");

      Add(parts, String(hom));
    od;

    if outfile <> false then
      WriteLine(fp, JoinStringsWithSeparator(parts, ","));
    fi;
    Print("\n");
  od;

  if outfile <> false then
    CloseStream(fp);
    Print("Results saved to ", outfile, "\n");
  fi;
end;