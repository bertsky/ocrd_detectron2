#!/bin/bash
# assumes the result artifact from make test has been downloaded already
find test-results/ -name test-result | {
    echo ---;
    echo layout: post;
    echo title: Test Results;
    echo ---;
    while read dataset; do
        dir=${dataset#test-results/};
        dir=${dir%/data/test-result};
        cat $dataset | {
            lastgrp=;
            echo "<table>";
            echo "<caption class=\"h1\">$dir</caption>";
            echo "<tr><th>model</th><th colspan=\"100\">pages</th></tr>";
            while IFS=: read path num; do
                path=${path#test/assets/};
                file=${path#$dir/data/};
                grp=$(dirname $file);
                page=${file#$grp/$grp-};
                page=${page%.xml};
                grp=${grp#OCR-D-SEG-};
                if test "x$grp" != "x$lastgrp"; then
                    test -n "$lastgrp" && echo '</tr>';
                    echo '<tr>';
                    echo "<th>$grp</th>";
                fi;
                echo "<td><img title=\"$num regions\" src=\"${path%.xml}.IMG-DEBUG.png\"/></td>";
                lastgrp="$grp";
            done;
            echo "</tr></table>";
        };
    done;
} > test-results.md
