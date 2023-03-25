#!/opt/local/bin/zsh

##################################
##################################
##################################

get_file_count () {
    echo $(echo "$(ls | wc -l)-1" | bc)
}
start_file_count=$(get_file_count)

##################################
##################################
##################################

expected_file_total=$1
get_percent () {
    echo $(echo "scale=4; ($(get_file_count)/$expected_file_total)*100" | bc)
}

##################################
##################################
##################################

get_current_timestamp () {
    echo $(date +%s)
}
start_timestamp=$(get_current_timestamp)

##################################
##################################
##################################

format_time () {
    d=$(($1/86400))
    h=$(($1%86400/3600))
    m=$(($1%3600/60))
    s=$(($1%60))
    echo "${d}d:${h}h:${m}m:${s}s"
}

##################################
##################################
##################################

count_percent () {
    while true
    do
        percent=$(get_percent)

        file_count=$(get_file_count)
        files_completed=$(echo "$file_count-$start_file_count" | bc)
        files_remaining=$(echo "$expected_file_total-$file_count" | bc)
        
        current_timestamp=$(get_current_timestamp)
        elapsed_time=$((current_timestamp-start_timestamp))

        if [[ $files_completed -gt 0 ]]; then
            average_time_per_file=$(echo "$elapsed_time/$files_completed" | bc)
            remaining_time=$(echo "$files_remaining*$average_time_per_file" | bc)
            projected_completion_timestamp=$(date -r $((current_timestamp+remaining_time)))

            echo $file_count/$expected_file_total files, ${percent:0:-2}% "complete, " "${average_time_per_file}s average per file, " $(format_time $remaining_time) "remaining, " "might be done on" $projected_completion_timestamp
        else
            echo $file_count/$expected_file_total files, ${percent:0:-2}% "complete, " $elapsed_time "seconds elapsed"
        fi

        sleep 1
    done
}

##################################
##################################
##################################

file_count=$(get_file_count)
start_percent=0
if [[ $file_count -gt 0 ]]; then
    start_percent=$(get_percent)
    count_percent
fi