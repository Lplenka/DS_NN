#Refer : https://www.mysqltutorial.org/mysql-create-table/

CREATE TABLE IF NOT EXISTS tasks (
    task_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    start_date DATE,
    due_date DATE,
    status TINYINT NOT NULL,
    priority TINYINT NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)

-- INSERT INTO table(c1,c2,...)
-- VALUES 
--    (v11,v12,...),
--    (v21,v22,...),
--     ...
--    (vnn,vn2,...);